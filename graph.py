"""LangGraph AI Agent — 入口，委派给 Supervisor"""
from agents.supervisor import build_supervisor


def build_graph():
    """向后兼容：返回 Supervisor 编译图"""
    return build_supervisor()
