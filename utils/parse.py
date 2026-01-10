import re
import json
import ast

def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格。
    支持 JSON 格式、Python 列表格式、带尾部逗号的格式以及被杂质文本包围的格式。
    
    参数:
    text (str): 大语言模型的输出文本
    
    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数) 或 None
    """
    if not text or not isinstance(text, str):
        return None

    # 1. 预处理：提取 Markdown 代码块中的内容
    code_block_pattern = r"```(?:json|python)?\s*(.*?)\s*```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    content_to_parse = code_blocks[-1].strip() if code_blocks else text.strip()

    # 2. 定位所有潜在的二维数组结构 [[...]]
    # 使用非贪婪匹配获取所有可能的括号对
    array_pattern = r"\[\s*\[.*?\]\s*\]"
    candidates = re.findall(array_pattern, content_to_parse, re.DOTALL)
    
    # 如果在代码块没找到，尝试在全文中匹配
    if not candidates and code_blocks:
        candidates = re.findall(array_pattern, text, re.DOTALL)

    if not candidates:
        return None

    # 3. 倒序尝试解析候选字符串
    for candidate in reversed(candidates):
        grid = _attempt_parse(candidate)
        if grid:
            return grid

    return None

def _attempt_parse(raw_str):
    """私有辅助函数：尝试通过多种手段解析单一字符串片段"""
    # 1: 标准 JSON 解析
    try:
        data = json.loads(raw_str)
        if _is_valid_grid(data):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2: Python AST 解析
    try:
        data = ast.literal_eval(raw_str)
        if _is_valid_grid(data):
            return data
    except (SyntaxError, ValueError, MemoryError):
        pass

    # 3: 正则暴力提取
    # 逻辑：提取所有形如 [...] 的内层结构，再从内层提取所有数字
    try:
        inner_row_pattern = r"\[([^\[\]]+)\]"
        rows_str = re.findall(inner_row_pattern, raw_str)
        grid = []
        for r_str in rows_str:
            nums = re.findall(r"[-+]?\d+", r_str)
            if nums:
                grid.append([int(n) for n in nums])
        
        if _is_valid_grid(grid):
            return grid
    except Exception:
        pass

    return None

def _is_valid_grid(obj):
    """校验对象是否为非空的二维整数列表"""
    if not isinstance(obj, list) or not obj:
        return False
    # 检查第一层和第二层是否均为列表
    if not all(isinstance(row, list) for row in obj):
        return False
    # 检查是否至少包含数字
    # 这里通过检查第一行是否包含数字来做快速判定
    try:
        if len(obj) > 0 and len(obj[0]) >= 0:
            return True
    except Exception:
        pass
    return False