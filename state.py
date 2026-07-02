"""DocQAState — LangGraph AI Agent 状态定义"""
from typing import TypedDict, Annotated, Optional
import operator
from langchain_core.messages import BaseMessage


class DocQAState(TypedDict):
    """LangGraph 全局状态，贯穿所有节点"""

    # === 对话消息（追加模式）===
    messages: Annotated[list[BaseMessage], operator.add]

    # === 用户输入 ===
    user_input: str

    # === 文档状态 ===
    current_document: Optional[str]
    pdf_file_path: Optional[str]
    documents_loaded: int
    total_chunks: int

    # === 路由 ===
    query_type: str
    command: str
    selected_model: str  # 当前选择的 LLM 模型

    # === 检索 ===
    retrieval_results: list[dict]
    context: str

    # === 输出 ===
    answer: str

    # ═══════════════════════════════════════
    # Agent 规划 & 反思（Phase 4 新增）
    # ═══════════════════════════════════════

    # 规划
    plan: list[dict]          # [{step_id, description, action_type, status}]
    current_step: int          # 当前执行到第几步
    max_plan_retries: int      # 反思重试上限

    # 反思
    reflection: str            # 反思评估结果
    needs_clarification: bool  # 是否需要追问用户
    clarification_question: str  # 追问内容

    # 执行日志（追加模式）
    execution_log: Annotated[list[str], operator.add]
    retry_count: Annotated[int, operator.add]

    # === 统计 ===
    questions_asked: Annotated[int, operator.add]
    concepts_learned: Annotated[int, operator.add]
    session_start: Optional[str]


def initial_state() -> DocQAState:
    """工厂函数"""
    from datetime import datetime
    return {
        "messages": [],
        "user_input": "",
        "current_document": None,
        "pdf_file_path": None,
        "documents_loaded": 0,
        "total_chunks": 0,
        "query_type": "",
        "command": "",
        "retrieval_results": [],
        "context": "",
        "answer": "",
        "plan": [],
        "current_step": 0,
        "max_plan_retries": 2,
        "reflection": "",
        "needs_clarification": False,
        "clarification_question": "",
        "execution_log": [],
        "retry_count": 0,
        "questions_asked": 0,
        "concepts_learned": 0,
        "session_start": datetime.now().isoformat(),
    }
