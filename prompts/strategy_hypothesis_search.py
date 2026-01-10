import json

def grid_to_matrix_str(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def construct_prompt(d):
    train_examples = d['train']
    test_input = d['test'][0]['input']

    # 1. System Prompt: 设定为严格的科学验证者
    system_content = (
        "You are an ARC Solution Architect.\n"
        "Your goal is to achieve 100% accuracy by filtering out incorrect assumptions.\n"
        "### STRICT PROTOCOL:\n"
        "1. **Brainstorm**: Propose 3 DISTINCT transformation rules (candidates) derived from the examples.\n"
        "2. **Verification (Crucial)**: Test EACH candidate against ALL training examples.\n"
        "   - If a candidate fails to explain ANY training example perfectly, mark it as [FAILED].\n"
        "   - If a candidate explains ALL examples, mark it as [PASSED].\n"
        "3. **Selection**: Select the SINGLE best [PASSED] candidate.\n"
        "4. **Execution**: Apply this winning rule to the Test Input.\n\n"
        "### Output Format:\n"
        "Provide the verification process, then the final result in a code block:\n"
        "```json\n"
        "[[...]]\n"
        "```"
    )

    # 2. User Prompt: 拼接训练样本
    user_content = "Find the correct rule through hypothesis verification.\n\n"

    for idx, example in enumerate(train_examples):
        user_content += f"--- Example {idx + 1} ---\n"
        user_content += f"Input:\n{grid_to_matrix_str(example['input'])}\n"
        user_content += f"Output:\n{grid_to_matrix_str(example['output'])}\n\n"

    # 3. 引导各种假设
    user_content += "--- Test Task ---\n"
    user_content += f"Input:\n{grid_to_matrix_str(test_input)}\n\n"
    
    user_content += (
        "Perform the Verification Protocol now:\n"
        "1. **Hypothesis Generation**:\n"
        "   - Candidate A (Visual focus): ...\n"
        "   - Candidate B (Object/Movement focus): ...\n"
        "   - Candidate C (Mathematical/Counting focus): ...\n\n"
        "2. **Verification Round**:\n"
        "   - Check A against Ex 1, 2, ... -> Result?\n"
        "   - Check B against Ex 1, 2, ... -> Result?\n"
        "   - Check C against Ex 1, 2, ... -> Result?\n\n"
        "3. **Final Selection & Execution**:\n"
        "   - Apply the best rule to the Test Input.\n"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages