"""RetrievalAgent — 多路召回 + 上下文合并"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agents.schemas import RetrievalState
from rag_pipeline import retrieve_advanced, build_context


def _search(state: RetrievalState) -> dict:
    """执行多路检索"""
    user_input = state.get("user_input", "")
    if not user_input:
        return {"retrieval_results": [], "context": "", "hit_count": 0}

    results = retrieve_advanced(user_input, k=5, enable_mqe=True, enable_hyde=True)
    context = build_context(results)

    return {
        "retrieval_results": results,
        "context": context,
        "hit_count": len(results),
        "messages": [HumanMessage(content=f"[Retrieval] 检索到 {len(results)} 条, 上下文 {len(context)} 字")],
    }


def build_retrieval_agent() -> StateGraph:
    builder = StateGraph(RetrievalState)
    builder.add_node("search", _search)
    builder.add_edge(START, "search")
    builder.add_edge("search", END)
    return builder.compile(checkpointer=MemorySaver())
