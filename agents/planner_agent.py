"""PlannerAgent — 任务拆解 + 工具选择"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agents.schemas import PlannerState
from planner import plan_task


def _plan(state: PlannerState) -> dict:
    """核心：LLM 拆解用户问题"""
    user_input = state.get("user_input", "")
    model_id = state.get("selected_model", "")

    try:
        result = plan_task(user_input, model_id=model_id)
    except Exception:
        result = {"needs_clarification": False, "clarification_question": "", "plan": [
            {"step_id": 1, "description": "直接回答", "action_type": "answer", "status": "pending"}]}

    for s in result["plan"]:
        s["status"] = "pending"

    return {
        "plan": result["plan"],
        "needs_clarification": result["needs_clarification"],
        "clarification_question": result["clarification_question"],
        "messages": [HumanMessage(content=f"[Planner] 拆解为 {len(result['plan'])} 步")]
    }


def build_planner_agent() -> StateGraph:
    builder = StateGraph(PlannerState)
    builder.add_node("plan", _plan)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", END)
    return builder.compile(checkpointer=MemorySaver())
