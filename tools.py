"""Agent 工具集 — Function Calling"""
import os
import json
from datetime import datetime
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """数学计算器。支持加减乘除、幂运算、括号。输入可以是自然语言，会自动提取表达式。"""
    import math
    import re

    # 从自然语言中提取数学表达式
    # 匹配数字和运算符的组合
    math_pattern = r'[\d\s\+\-\*\/\(\)\.\^\%]+'
    matches = re.findall(math_pattern, expression)
    if matches:
        # 取最长匹配
        expr = max(matches, key=len).strip()
        # 如果提取的表达式看起来合理（至少有数字和运算符）
        if re.search(r'\d', expr) and re.search(r'[\+\-\*\/]', expr):
            expression = expr

    safe_dict = {
        "math": math, "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "int": int, "float": float,
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e} (表达式: {expression})"


@tool
def web_search(query: str) -> str:
    """联网搜索最新信息。用于时效性问题、超出文档范围的知识。输入搜索关键词。"""
    try:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return "⚠️ Tavily API Key 未配置"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "未找到相关信息"

        parts = []
        for r in results:
            parts.append(f"📌 {r.get('title', '')}\n{r.get('content', '')[:300]}")
        return "\n\n".join(parts)
    except ImportError:
        return "⚠️ 请安装 tavily-python: pip install tavily-python"
    except Exception as e:
        return f"搜索失败: {e}"


@tool
def read_notes(query: str = "") -> str:
    """读取已保存的学习笔记。query 可选，用于关键词过滤；不匹配时返回全部笔记。"""
    notes_file = os.path.join(os.path.dirname(__file__), "learning_notes.json")
    if not os.path.exists(notes_file):
        return "📝 暂无笔记（文件不存在）"

    try:
        with open(notes_file, "r", encoding="utf-8") as f:
            notes = json.load(f)
    except Exception as e:
        return f"📝 读取笔记失败: {e}"

    if not notes:
        return "📝 暂无笔记"

    # 如果有关键词，尝试过滤；过滤为空则返回全部
    if query and len(query) > 2:
        filtered = [n for n in notes if query.lower() in n.get("content", "").lower()]
        if filtered:
            notes = filtered

    parts = []
    for i, n in enumerate(notes[-5:], 1):
        ts = n.get("timestamp", "")[:16]
        parts.append(f"{i}. [{ts}] {n['content'][:200]}")

    return f"📝 **学习笔记**（共 {len(notes)} 条）:\n" + "\n".join(parts)


@tool
def get_current_time(_: str = "") -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} (星期{['一','二','三','四','五','六','日'][now.weekday()]})"


# 所有可用工具
ALL_TOOLS = [calculator, web_search, read_notes, get_current_time]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
