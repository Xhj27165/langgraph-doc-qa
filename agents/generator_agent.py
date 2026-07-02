"""GeneratorAgent — 流式生成 + 引用标注"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.config import get_stream_writer
from agents.schemas import GeneratorState
from rag_pipeline import get_llm


def _generate(state: GeneratorState) -> dict:
    """流式生成答案 + 引用标注"""
    writer = get_stream_writer()
    user_input = state.get("user_input", "")
    context = state.get("context", "")

    if not context:
        return {"answer": "🤔 上下文为空，无法生成答案", "messages": [AIMessage(content="上下文为空")]}

    model_id = state.get("selected_model", "")
    llm = get_llm(model_id) if model_id else get_llm()

    system_prompt = "你是专业 AI Agent。严格基于上下文回答。不确定则坦诚说明。中文回答。"
    user_prompt = f"【问题】{user_input}\n\n【上下文】\n{context}\n\n请回答："

    writer("[Generator] 正在生成...\n")
    full_response = ""
    for chunk in llm.stream([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]):
        content = chunk.content if hasattr(chunk, "content") else ""
        if content:
            full_response += content
            writer(content)

    # 引用标注
    results = state.get("retrieval_results", [])
    if results:
        full_response += "\n\n📚 **参考来源**\n"
        seen = set()
        for i, r in enumerate(results, 1):
            src = r.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                full_response += f"• [{i}] {src} (相似度: {r.get('score', 0):.3f})\n"

    return {
        "answer": full_response,
        "messages": [AIMessage(content=full_response), HumanMessage(content=f"[Generator] 生成 {len(full_response)} 字")],
    }


def build_generator_agent() -> StateGraph:
    builder = StateGraph(GeneratorState)
    builder.add_node("generate", _generate)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=MemorySaver())
