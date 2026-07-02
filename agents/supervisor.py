"""SupervisorAgent — 多智能体协调调度器 (v2.0)

v2.0: 集成 agent_config.yaml + ToolCenter + RAGAS 深度评估
"""
import os
import sqlite3
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.config import get_stream_writer

from state import DocQAState
from rag_pipeline import (
    load_and_index_pdf, retrieve_advanced, build_context, get_llm,
)
from planner import plan_task
from reflector import reflect
from evaluator import create_eval_entry, append_eval

# v2.0: 从配置加载参数
try:
    from config_loader import get_config
    _config = get_config()
    _MAX_RETRIES = _config.evaluator_max_retries
    _QUALITY_THRESHOLD = _config.evaluator_quality_threshold
    _RAGAS_ENABLED = _config.ragas_enabled
except Exception:
    _MAX_RETRIES = 2
    _QUALITY_THRESHOLD = 3
    _RAGAS_ENABLED = True

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints.db")
_conn = None

# v2.1: Request timing tracker (module-level, keyed by thread_id for now)
_request_timers: dict[str, float] = {}
_request_tokens: dict[str, dict] = {}  # accumulated token counts per request

def _get_checkpointer() -> SqliteSaver:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(_conn)


# ═══════════════════════════════════════════
# 基础节点（保留简单路径）
# ═══════════════════════════════════════════

def record_input(state: DocQAState) -> dict:
    ui = state.get("user_input", "").strip()
    if not ui:
        return {}
    # Track request start time for latency measurement
    import time
    _request_timers["current"] = time.time()
    _request_tokens["current"] = {"input": 0, "output": 0}
    return {
        "messages": [HumanMessage(content=ui)],
        "answer": "", "context": "", "retrieval_results": [],
        "query_type": "", "plan": [], "current_step": 0, "reflection": "",
    }

def classify_intent(state: DocQAState) -> dict:
    cmd = state.get("command", "").strip()
    if cmd:
        return {"query_type": cmd, "command": ""}
    ui = state.get("user_input", "").strip()
    if not ui:
        return {"query_type": "qa"}
    if any(k in ui for k in ["加载", "上传"]): return {"query_type": "load_doc"}
    if any(k in ui for k in ["回顾之前","回顾所学","回顾学过","学了什么","所有学过","全部学过","总结学过","回顾所有"]): return {"query_type": "recall"}
    if ui.strip() in ["回顾","总结"]: return {"query_type": "recall"}
    if any(ui.startswith(k) or f" {k}" in ui for k in ["笔记","记录"]): return {"query_type": "notes"}
    if any(k in ui for k in ["报告","统计","进度"]): return {"query_type": "report"}
    return {"query_type": "qa"}

def index_document(state: DocQAState) -> dict:
    pdf_path = state.get("pdf_file_path")
    if not pdf_path: return {"answer": "[Error] Please upload a PDF file first"}
    if not os.path.exists(pdf_path): return {"answer": f"[Error] File not found: {pdf_path}"}
    force = any(k in state.get("user_input","") for k in ["重新","强制"])
    result = load_and_index_pdf(pdf_path, force_reload=force)
    return {"current_document": result["document"] if result["success"] else state.get("current_document"),
            "documents_loaded": state.get("documents_loaded",0)+(1 if result["success"] else 0),
            "total_chunks": result.get("chunks",0),
            "answer": f"[OK] {result['message']}" if result["success"] else f"[Error] {result['message']}"}

def recall_memory(state: DocQAState) -> dict:
    msgs = state.get("messages",[])
    qa_pairs, uq = [], None
    for m in msgs:
        if isinstance(m, HumanMessage): uq = m.content
        elif isinstance(m, AIMessage) and uq:
            a = m.content.strip()
            if a and not a.startswith("[OK]") and not a.startswith("[Error]") and not a.startswith("["):
                qa_pairs.append({"question":uq,"answer":a[:300]})
            uq = None
    if not qa_pairs: return {"answer":"📝 暂无问答记录"}
    rpt = f"🧠 **学习历程回顾**（共 {state.get('questions_asked',len(qa_pairs))} 次）\n\n"
    for i,p in enumerate(qa_pairs[-10:],1):
        rpt += f"### Q{i}: {p['question'][:100]}\n> {p['answer'][:150]}...\n\n"
    return {"answer":rpt}

def add_note(state: DocQAState) -> dict:
    import json
    from datetime import datetime
    ui = state.get("user_input","")
    if not ui.strip(): return {"answer":"[Error] Note content is empty"}
    note = {"timestamp":datetime.now().isoformat(),"content":ui,"document":state.get("current_document","")}
    nf = os.path.join(os.path.dirname(__file__),"..","learning_notes.json")
    notes = []
    if os.path.exists(nf):
        try:
            with open(nf,"r",encoding="utf-8") as f: notes = json.load(f)
        except: pass
    notes.append(note)
    with open(nf,"w",encoding="utf-8") as f: json.dump(notes,f,ensure_ascii=False,indent=2)
    return {"answer":f"[OK] Note saved! ({len(notes)} total)\n\n> {ui[:200]}","concepts_learned":1}

def generate_report(state: DocQAState) -> dict:
    import json
    from datetime import datetime
    ss = state.get("session_start","")
    dur = (datetime.now()-datetime.fromisoformat(ss)).total_seconds() if ss else 0
    msgs = state.get("messages",[])
    qc = sum(1 for m in msgs if isinstance(m,HumanMessage) and m.content.strip() and "加载" not in m.content and "回顾" not in m.content)
    nf = os.path.join(os.path.dirname(__file__),"..","learning_notes.json")
    nc = 0
    if os.path.exists(nf):
        try:
            with open(nf,"r",encoding="utf-8") as f: nc = len(json.load(f))
        except: pass
    from evaluator import get_summary, get_cost_alert
    es = get_summary()
    report = (f"📊 **学习报告**\n\n| 指标 | 值 |\n|------|----|\n"
              f"| 会话时长 | {dur:.0f}s |\n| 当前文档 | {state.get('current_document','未加载')} |\n"
              f"| 提问次数 | {state.get('questions_asked',qc)} |\n| 学习笔记 | {nc} 条 |\n"
              f"| 重试次数 | {state.get('retry_count',0)} |\n")
    if es["total"]>0:
        report += (f"\n📈 **Agent 评估**（累计 {es['total']} 次）\n\n"
                   f"| 指标 | 值 |\n|------|----|\n"
                   f"| 成功率 | {es['success_rate']}% |\n| 平均质量评分 | {es['avg_score']}/5 |\n"
                   f"| 累计 Token | {es['total_tokens']:,} |\n| 累计成本 | ¥{es['total_cost']:.4f} |\n")
        # RAGAS 统计
        if es.get("ragas_count", 0) > 0:
            report += f"| RAGAS 综合评分 | {es['avg_ragas']:.3f} |\n| RAGAS 评估次数 | {es['ragas_count']} |\n"
    # 成本预警
    alert = get_cost_alert()
    if alert:
        report += f"\n{alert}\n"
    return {"answer":report}


# ═══════════════════════════════════════════
# Multi-Agent 协调节点 (v2.0)
# ═══════════════════════════════════════════

def _agent_log(agent_name: str, msg: str):
    """记录 Agent 协作日志"""
    from langgraph.config import get_stream_writer
    w = get_stream_writer()
    w(f"🤖 **[{agent_name}]** {msg}\n")


def planner_node(state: DocQAState) -> dict:
    """PlannerAgent: 拆解任务"""
    _agent_log("Planner", "分析问题，制定计划...")
    user_input = state.get("user_input", "")
    model_id = state.get("selected_model", "")
    try:
        result = plan_task(user_input, model_id=model_id)
    except:
        result = {"needs_clarification":False,"clarification_question":"","plan":[{"step_id":1,"description":"直接回答","action_type":"answer","status":"pending"}]}

    for s in result["plan"]: s["status"] = "pending"
    steps_desc = "\n".join(f"  {s['step_id']}. [{s.get('action_type','?')}] {s['description']}" for s in result["plan"])
    _agent_log("Planner", f"计划 ({len(result['plan'])} 步):\n{steps_desc}")

    if result["needs_clarification"]:
        return {"needs_clarification":True, "clarification_question":result["clarification_question"], "plan":[]}
    return {"plan": result["plan"], "needs_clarification":False}


def retrieval_node(state: DocQAState) -> dict:
    """RetrievalAgent: 多路检索所有计划步骤"""
    _agent_log("Retrieval", "执行多路检索...")
    user_input = state.get("user_input", "")
    plan = state.get("plan", [])
    all_parts = []

    # 遍历计划中的检索步骤
    for step in plan:
        if step.get("action_type") == "retrieve":
            query = f"{step['description']} {user_input}"
            results = retrieve_advanced(query, k=3, enable_mqe=True, enable_hyde=False)
            for r in results:
                all_parts.append(r["content"])
            step["status"] = "done"
            _agent_log("Retrieval", f"  检索: {step['description'][:50]} → {len(results)} 条")

    # 合并去重
    if all_parts:
        unique = list(dict.fromkeys(all_parts))
        context = "\n\n".join(unique)
    else:
        results = retrieve_advanced(user_input, k=5, enable_mqe=True, enable_hyde=True)
        context = build_context(results)

    if len(context) > 3000:
        context = context[:3000] + "..."

    _agent_log("Retrieval", f"上下文 {len(context)} 字, {len(all_parts)} 片段")
    return {"context": context, "retrieval_results": [{"content": c, "source": state.get("current_document","doc"), "score": 0} for c in all_parts[:5]],
            "plan": plan}


def executor_node(state: DocQAState) -> dict:
    """执行工具调用步骤 (v2.0: 使用 ToolCenter)"""
    plan = state.get("plan", [])
    user_input = state.get("user_input", "")

    # v2.0: 优先使用 ToolCenter
    try:
        from tool_center import get_tool_center
        center = get_tool_center()
    except Exception:
        from tools import TOOLS_BY_NAME as _fallback
        center = None

    for step in plan:
        if step.get("action_type") == "tool_call" and step.get("status") != "done":
            tool_name = step.get("tool_name", "")
            _agent_log("Executor", f"🔧 {tool_name}: {step['description'][:60]}")
            try:
                if center:
                    result = center.invoke(tool_name, step.get("description", user_input))
                else:
                    from tools import TOOLS_BY_NAME
                    tool = TOOLS_BY_NAME.get(tool_name)
                    result = str(tool.invoke(step.get("description", user_input))) if tool else f"工具 {tool_name} 不存在"
                _agent_log("Executor", f"   ✅ {str(result)[:120]}")
            except Exception as e:
                _agent_log("Executor", f"   ❌ {e}")
            step["status"] = "done"
    return {"plan": plan}


def generator_node(state: DocQAState) -> dict:
    """GeneratorAgent: 流式生成答案"""
    _agent_log("Generator", "综合生成答案...")
    from langgraph.config import get_stream_writer
    writer = get_stream_writer()

    user_input = state.get("user_input", "")
    context = state.get("context", "")
    model_id = state.get("selected_model", "")

    if not context:
        return {"answer": "🤔 上下文为空"}

    llm = get_llm(model_id) if model_id else get_llm()
    messages = [
        {"role":"system","content":"你是专业 AI Agent。严格基于上下文回答。不确定则坦诚说明。中文。"},
        {"role":"user","content":f"【问题】{user_input}\n\n【上下文】\n{context}\n\n请回答："},
    ]

    full = ""
    last_chunk = None
    for chunk in llm.stream(messages):
        c = chunk.content if hasattr(chunk,"content") else ""
        if c: full += c; writer(c)
        last_chunk = chunk

    # Capture token usage from streaming response
    from tracer import extract_token_usage
    gen_in, gen_out = extract_token_usage(last_chunk) if last_chunk else (0, 0)
    if "current" in _request_tokens:
        _request_tokens["current"]["input"] += gen_in
        _request_tokens["current"]["output"] += gen_out

    # 引用
    results = state.get("retrieval_results", [])
    if results:
        full += "\n\n📚 **参考来源**\n"
        seen = set()
        for i, r in enumerate(results, 1):
            src = r.get("source","unknown")
            if src not in seen: seen.add(src); full += f"• [{i}] {src}\n"

    _agent_log("Generator", f"生成 {len(full)} 字")
    return {"answer": full, "questions_asked": 1, "messages": [AIMessage(content=full)]}


def evaluator_node(state: DocQAState) -> dict:
    """EvaluatorAgent: 质量评估 + RAGAS 深度评估 + 追踪记录 (v2.1)"""
    import time
    _agent_log("Evaluator", "评估答案质量...")
    ui, ans, ctx = state.get("user_input",""), state.get("answer",""), state.get("context","")
    model_id = state.get("selected_model","")

    # Calculate elapsed time since request started
    start_time = _request_timers.pop("current", time.time())
    elapsed_ms = int((time.time() - start_time) * 1000)

    if not ans or len(ans) < 20:
        return {"reflection":"incomplete","retry_count":1,"plan":[], "current_step":0}

    result = reflect(ui, ans, ctx, model_id=model_id)

    # RAGAS 深度评估（按需）
    ragas_result = None
    if _RAGAS_ENABLED and len(ans) > 50:
        try:
            from ragas_eval import run_ragas_evaluation, format_ragas_report
            retrieval_docs = state.get("retrieval_results", [])
            ragas_result = run_ragas_evaluation(
                question=ui, answer=ans, context=ctx,
                retrieval_results=retrieval_docs, model_id=model_id,
            )
            _agent_log("Evaluator",
                       f"RAGAS: overall={ragas_result['overall_score']:.3f} "
                       f"({ragas_result['verdict']}) "
                       f"[{ragas_result['metrics_computed']} metrics]")
        except Exception as e:
            _agent_log("Evaluator", f"RAGAS 评估失败: {e}")

    # Get accumulated token counts from generator + estimate evaluation tokens
    tokens = _request_tokens.pop("current", {"input": 0, "output": 0})
    # Estimate: reflector (1 LLM call) + RAGAS (up to 4 LLM calls)
    eval_tokens = 200  # rough estimate per LLM call for evaluation prompts
    total_input = tokens.get("input", 0) + eval_tokens
    total_output = tokens.get("output", 0) + eval_tokens

    # 记录评估
    retrieval_docs = [r.get("content","") for r in state.get("retrieval_results",[])]
    entry = create_eval_entry(ui, ans, ctx, retrieval_docs, model_id,
                              input_tokens=total_input,
                              output_tokens=total_output,
                              elapsed_ms=elapsed_ms,
                              reflection_score=result.get("score",3),
                              success=result.get("quality")=="good",
                              ragas_result=ragas_result)
    append_eval(entry)

    # Save execution trace
    try:
        from tracer import RequestTrace, save_trace
        trace = RequestTrace(question=ui, model_id=model_id)
        trace.add_node("planner_node", input_preview=ui,
                       output_preview=str(state.get("plan", []))[:100], duration_ms=0)
        trace.add_node("retrieval_node",
                       input_preview=ui[:80], output_preview=ctx[:100],
                       duration_ms=0)
        trace.add_node("generator_node",
                       input_preview=ctx[:100], output_preview=ans[:150],
                       input_tokens=tokens.get("input", 0),
                       output_tokens=tokens.get("output", 0),
                       duration_ms=max(elapsed_ms // 2, 100))
        trace.add_node("evaluator_node",
                       input_preview=ans[:100], output_preview=result.get("quality", "?"),
                       input_tokens=eval_tokens, output_tokens=eval_tokens,
                       duration_ms=max(elapsed_ms // 4, 50))
        trace.total_duration_ms = elapsed_ms
        trace.total_input_tokens = total_input
        trace.total_output_tokens = total_output
        save_trace(trace)
    except Exception as e:
        _agent_log("Evaluator", f"Trace save failed: {e}")

    retry = result.get("should_retry", False)
    if retry and state.get("retry_count",0) < _MAX_RETRIES:
        _agent_log("Evaluator", f"⚠️ 质量不足(score={result['score']}/5), 重试")
        return {"reflection":result.get("quality"),"retry_count":1,"plan":[],"current_step":0,
                "execution_log":[f"评估: {result.get('quality')} score={result['score']}/5 重试"]}

    _agent_log("Evaluator", f"✅ {result.get('quality')} (score={result['score']}/5)")
    return {"reflection":result.get("quality","good"),
            "execution_log":[f"评估: {result.get('quality')} score={result['score']}/5"]}


# ═══════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════

def route_by_intent(state: DocQAState) -> Literal["index_document","planner_node","recall_memory","add_note","generate_report"]:
    q = state.get("query_type","qa")
    return {"load_doc":"index_document","qa":"planner_node","recall":"recall_memory","notes":"add_note","report":"generate_report"}.get(q,"planner_node")

def route_after_eval(state: DocQAState) -> Literal["retrieval_node","__end__"]:
    r = state.get("reflection","")
    if r in ("incomplete","hallucination","off_topic") and state.get("retry_count",0) < _MAX_RETRIES:
        return "retrieval_node"
    return "__end__"


# ═══════════════════════════════════════════
# 构建 Supervisor Graph
# ═══════════════════════════════════════════

def build_supervisor():
    builder = StateGraph(DocQAState)

    # 基础节点
    builder.add_node("record_input", record_input)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("index_document", index_document)
    builder.add_node("recall_memory", recall_memory)
    builder.add_node("add_note", add_note)
    builder.add_node("generate_report", generate_report)

    # 多 Agent 节点
    builder.add_node("planner_node", planner_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("executor_node", executor_node)
    builder.add_node("generator_node", generator_node)
    builder.add_node("evaluator_node", evaluator_node)

    # 连线
    builder.add_edge(START, "record_input")
    builder.add_edge("record_input", "classify_intent")
    builder.add_conditional_edges("classify_intent", route_by_intent, {
        "index_document":"index_document","planner_node":"planner_node",
        "recall_memory":"recall_memory","add_note":"add_note","generate_report":"generate_report",
    })
    builder.add_edge("index_document", END)
    builder.add_edge("recall_memory", END)
    builder.add_edge("add_note", END)
    builder.add_edge("generate_report", END)

    # Agent 循环: planner → executor → retrieval → generator → evaluator → (retry or END)
    builder.add_edge("planner_node", "executor_node")
    builder.add_edge("executor_node", "retrieval_node")
    builder.add_edge("retrieval_node", "generator_node")
    builder.add_edge("generator_node", "evaluator_node")
    builder.add_conditional_edges("evaluator_node", route_after_eval, {
        "retrieval_node":"retrieval_node","__end__":END,
    })

    return builder.compile(checkpointer=_get_checkpointer())
