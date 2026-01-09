import re
import json

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