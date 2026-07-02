"""各 Sub-Agent 的独立 State Schema"""
from typing import TypedDict, Annotated, Optional
import operator
from langchain_core.messages import BaseMessage


class PlannerState(TypedDict):
    """PlannerAgent 内部状态"""
    user_input: str
    messages: Annotated[list[BaseMessage], operator.add]
    plan: list[dict]           # [{step_id, description, action_type, tool_name, status}]
    needs_clarification: bool
    clarification_question: str


class RetrievalState(TypedDict):
    """RetrievalAgent 内部状态"""
    user_input: str
    sub_queries: list[str]     # MQE 扩展查询列表
    retrieval_results: list[dict]
    context: str               # 合并后的上下文
    hit_count: int             # 检索命中数


class GeneratorState(TypedDict):
    """GeneratorAgent 内部状态"""
    user_input: str
    context: str
    retrieval_results: list[dict]
    answer: str
    stream_tokens: list[str]


class EvaluatorState(TypedDict):
    """EvaluatorAgent 内部状态"""
    user_input: str
    answer: str
    context: str
    retrieval_results: list[dict]
    quality: str              # good/incomplete/hallucination/off_topic
    score: int                # 1-5
    issues: list[str]
    should_retry: bool
    retry_suggestion: str
