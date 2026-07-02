"""EvaluatorAgent — 质量评分 + 幻觉检测 + 重试决策"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agents.schemas import EvaluatorState
from reflector import reflect
from evaluator import create_eval_entry, append_eval


def _evaluate(state: EvaluatorState) -> dict:
    """LLM-as-Judge 评估答案 + 记录指标"""
    user_input = state.get("user_input", "")
    answer = state.get("answer", "")
    context = state.get("context", "")

    if not answer or len(answer) < 20:
        return {"quality": "incomplete", "score": 1, "should_retry": True,
                "retry_suggestion": "答案过短，重新检索并生成", "issues": ["答案过短"]}

    model_id = state.get("selected_model", "")
    result = reflect(user_input, answer, context, model_id=model_id)

    # 记录评估日志
    retrieval_docs = [r.get("content", "") for r in state.get("retrieval_results", [])]
    entry = create_eval_entry(
        user_input=user_input, answer=answer, context=context,
        retrieval_docs=retrieval_docs, model_id=model_id,
        reflection_score=result.get("score", 3),
        success=result.get("quality") == "good",
    )
    append_eval(entry)

    return {
        "quality": result.get("quality", "good"),
        "score": result.get("score", 3),
        "issues": result.get("issues", []),
        "should_retry": result.get("should_retry", False),
        "retry_suggestion": result.get("retry_suggestion", ""),
        "messages": [HumanMessage(content=f"[Evaluator] score={result.get('score', 3)}/5 quality={result.get('quality', '?')}")],
    }


def build_evaluator_agent() -> StateGraph:
    builder = StateGraph(EvaluatorState)
    builder.add_node("evaluate", _evaluate)
    builder.add_edge(START, "evaluate")
    builder.add_edge("evaluate", END)
    return builder.compile(checkpointer=MemorySaver())
