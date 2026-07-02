"""Gradio 6.x Web UI — 智能文档问答助手"""
from typing import List
import os
import gradio as gr

from graph import build_graph
from state import initial_state


_app_graph = None
_thread_id = "default"
_selected_model = "deepseek-chat"


def set_model(model_id: str):
    global _selected_model
    _selected_model = model_id


def get_graph():
    global _app_graph
    if _app_graph is None:
        _app_graph = build_graph()
    return _app_graph


def init_assistant(user_id: str) -> str:
    global _thread_id
    if not user_id:
        user_id = "web_user"
    _thread_id = user_id
    get_graph()
    try:
        from tool_center import init_tool_center
        tc = init_tool_center()
        tools = tc.list_enabled()
        tool_list = ", ".join(e.name for e in tools)
        return f"✅ 助手已初始化 (用户: {user_id})\n📦 LangGraph StateGraph 就绪\n🔧 已加载工具: {tool_list}"
    except Exception:
        return f"✅ 助手已初始化 (用户: {user_id})\n📦 LangGraph StateGraph 就绪"


def load_pdf(pdf_file) -> str:
    if pdf_file is None:
        return "❌ 请上传 PDF 文件"
    graph = get_graph()
    pdf_path = pdf_file.name
    result = graph.invoke(
        {
            "user_input": f"加载文档: {pdf_path}",
            "pdf_file_path": pdf_path,
            "command": "load_doc",
            "selected_model": _selected_model,
        },
        {"configurable": {"thread_id": _thread_id}},
    )
    return result.get("answer", "❌ 加载失败")


def chat(message: str, history: List[dict]):
    graph = get_graph()
    if not message.strip():
        yield "", history
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    last_answer = ""
    try:
        for chunk in graph.stream(
            {"user_input": message, "selected_model": _selected_model},
            {"configurable": {"thread_id": _thread_id}},
            stream_mode="custom",
        ):
            if isinstance(chunk, str):
                last_answer += chunk
                history[-1]["content"] = last_answer
                yield "", history

        result = graph.get_state({"configurable": {"thread_id": _thread_id}})
        if result and result.values:
            final_answer = result.values.get("answer", last_answer)
            if final_answer and final_answer != last_answer:
                history[-1]["content"] = final_answer
                yield "", history

    except Exception as e:
        history[-1]["content"] = f"❌ 错误: {str(e)}"
        yield "", history


def add_note_ui(note_content: str) -> str:
    if not note_content.strip():
        return "❌ 笔记内容不能为空"
    graph = get_graph()
    result = graph.invoke(
        {"user_input": note_content, "command": "notes", "selected_model": _selected_model},
        {"configurable": {"thread_id": _thread_id}},
    )
    return result.get("answer", "❌ 保存失败")


def get_stats_ui() -> str:
    from evaluator import get_summary
    graph = get_graph()
    result = graph.invoke(
        {"user_input": "统计", "command": "report", "selected_model": _selected_model},
        {"configurable": {"thread_id": _thread_id}},
    )
    basic = result.get("answer", "暂无统计")
    summary = get_summary()
    if summary["total"] > 0:
        eval_str = (
            f"\n\n📈 **Agent 评估指标**（累计 {summary['total']} 次）\n\n"
            f"| 指标 | 值 |\n|------|----|\n"
            f"| 成功率 | {summary['success_rate']}% |\n"
            f"| 平均质量评分 | {summary['avg_score']}/5 |\n"
            f"| 平均检索命中率 | {summary['avg_hit_rate']:.1%} |\n"
            f"| 累计 Token | {summary['total_tokens']:,} |\n"
            f"| 累计成本 | ¥{summary['total_cost']:.4f} |\n"
            f"| 使用模型 | {', '.join(summary['models_used'])} |\n"
        )
        if summary.get("ragas_count", 0) > 0:
            eval_str += f"| RAGAS 综合评分 | {summary['avg_ragas']:.3f} |\n"
            eval_str += f"| RAGAS 评估次数 | {summary['ragas_count']} |\n"
        return basic + eval_str
    return basic


def get_tool_center_stats() -> str:
    try:
        from tool_center import get_tool_center
        center = get_tool_center()
        stats = center.get_stats()
        lines = ["## 🔧 工具中心状态\n"]
        lines.append("| 工具 | 分类 | 状态 | 调用次数 | 成功率 | 平均延迟 |")
        lines.append("|------|------|------|----------|--------|----------|")
        for entry in center.list_all():
            s = stats.get(entry.name, {})
            status = "✅" if entry.enabled else "🔒"
            lines.append(
                f"| {entry.name} | {entry.category} | {status} | "
                f"{s.get('calls', 0)} | {s.get('success_rate', 0):.1%} | "
                f"{s.get('avg_latency_ms', 0):.0f}ms |"
            )
        lines.append(f"\n**分类统计**: {center.get_categories()}")
        from evaluator import get_cost_alert
        alert = get_cost_alert()
        if alert:
            lines.append(f"\n{alert}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ ToolCenter 未初始化: {e}"


def run_ragas_on_last() -> str:
    graph = get_graph()
    state = graph.get_state({"configurable": {"thread_id": _thread_id}})
    if not state or not state.values:
        return "❌ 暂无对话状态，请先在智能问答中提问"
    ans = state.values.get("answer", "")
    ctx = state.values.get("context", "")
    retrieval = state.values.get("retrieval_results", [])
    ui = state.values.get("user_input", "")
    if not ctx or not retrieval:
        return "⚠️ 上一次非文档问答操作，无检索数据。请在智能问答中先提问。"
    if not ans or len(ans) < 20:
        return "⚠️ 回答太短，请先进行一次完整问答。"
    try:
        from ragas_eval import run_ragas_evaluation, format_ragas_report
        result = run_ragas_evaluation(
            question=ui, answer=ans, context=ctx,
            retrieval_results=retrieval, model_id=_selected_model,
        )
        return format_ragas_report(result)
    except Exception as e:
        return f"❌ RAGAS 评估失败: {e}"


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="📚 智能文档问答助手 (LangGraph)") as demo:
        gr.Markdown("""
        # 📚 智能文档问答助手
        <p style="color: #666;">基于 <b>LangGraph</b> + <b>Qdrant</b> + <b>DashScope</b> 的多智能体文档问答系统</p>

        * 📄 上传 PDF → 自动分块 → 向量化存储
        * 💬 5-Agent 协作问答（Planner → Executor → Retrieval → Generator → Evaluator）
        * 📊 RAGAS 深度评估（忠实度 / 相关性 / 精确度 / 召回率）
        * 🧠 学习历程回顾 | 📝 学习笔记 | 🔧 工具中心
        """)

        with gr.Tab("🏠 开始使用"):
            with gr.Row():
                user_id_input = gr.Textbox(
                    label="用户 ID",
                    placeholder="输入用户 ID（可选）",
                    value="web_user",
                )
                model_selector = gr.Dropdown(
                    label="🤖 LLM 模型",
                    choices=[],
                    value=_selected_model,
                    interactive=True,
                )
                init_btn = gr.Button("初始化助手", variant="primary")
            init_output = gr.Textbox(label="初始化状态", interactive=False, lines=5)

            def refresh_models():
                from model_factory import get_available_models
                models = get_available_models()
                choices = [m["id"] for m in models]
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

            demo.load(refresh_models, outputs=[model_selector])
            model_selector.change(set_model, inputs=[model_selector])
            init_btn.click(init_assistant, inputs=[user_id_input], outputs=[init_output])

            gr.Markdown("### 📄 加载 PDF 文档")
            pdf_upload = gr.File(
                label="上传 PDF 文件",
                file_types=[".pdf"],
                type="filepath",
            )
            load_btn = gr.Button("加载文档", variant="primary")
            load_output = gr.Textbox(label="加载状态", interactive=False)
            load_btn.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])

        with gr.Tab("💬 智能问答"):
            gr.Markdown("### 向文档提问 或 回顾学习历程")
            chatbot = gr.Chatbot(label="对话历史", height=400)

            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：什么是大语言模型？  或  回顾我之前学过的所有内容",
                    scale=4,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "什么是大语言模型？",
                    "Transformer 架构有哪些核心组件？",
                    "回顾我之前学过的所有内容",
                    "我学了什么？",
                ],
                inputs=msg_input,
            )

            msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

        with gr.Tab("📝 学习笔记"):
            gr.Markdown("### 记录学习心得")
            note_content = gr.Textbox(
                label="笔记内容",
                placeholder="输入你的学习笔记...",
                lines=4,
            )
            note_btn = gr.Button("保存笔记", variant="primary")
            note_output = gr.Textbox(label="保存状态", interactive=False)
            note_btn.click(add_note_ui, inputs=[note_content], outputs=[note_output])

        with gr.Tab("📊 学习统计"):
            gr.Markdown("### 查看学习进度和统计")
            stats_btn = gr.Button("刷新统计", variant="primary")
            stats_output = gr.Markdown()
            stats_btn.click(get_stats_ui, outputs=[stats_output])

        with gr.Tab("🔧 工具中心"):
            gr.Markdown("### 工具状态 & 使用统计")
            tc_refresh_btn = gr.Button("刷新工具状态", variant="primary")
            tc_output = gr.Markdown()
            tc_refresh_btn.click(get_tool_center_stats, outputs=[tc_output])

        with gr.Tab("📊 RAGAS 评估"):
            gr.Markdown("### RAGAS 深度评估")
            gr.Markdown("对最近一次回答进行 4 维评估：**忠实度** | **答案相关性** | **上下文精确度** | **上下文召回率**")
            ragas_btn = gr.Button("运行 RAGAS 评估", variant="primary")
            ragas_output = gr.Markdown()
            ragas_btn.click(run_ragas_on_last, outputs=[ragas_output])

        # ═══════════════════════════════════════
        # Phase 10: Monitoring & Visualization
        # ═══════════════════════════════════════

        with gr.Tab("🔍 工作流图谱"):
            gr.Markdown("### LangGraph 工作流拓扑")
            gr.Markdown("当前系统的 StateGraph 结构，包含所有 Agent 节点和路由关系。")

            def render_graph():
                try:
                    import base64
                    graph = get_graph()
                    from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle, NodeStyles

                    # PNG rendering: fixed colors regardless of dark/light mode
                    png_bytes = graph.get_graph().draw_mermaid_png(
                        curve_style=CurveStyle.LINEAR,
                        node_colors=NodeStyles(
                            default="#eff6ff",   # light blue bg
                            first="#dbeafe",     # START: slightly darker blue
                            last="#dcfce7",      # END: light green
                        ),
                        draw_method=MermaidDrawMethod.API,
                        background_color="white",
                        padding=10,
                    )
                    b64 = base64.b64encode(png_bytes).decode("utf-8")
                    nodes_count = len(graph.get_graph().nodes)
                    edges_count = len(graph.get_graph().edges)
                    stats = f"\n\n> {nodes_count} nodes | {edges_count} edges | Multi-Agent Pipeline\n\n"
                    return stats + f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #e2e8f0;border-radius:8px;" alt="Workflow Graph" />'
                except Exception as e:
                    return f"Failed to render graph: {e}"

            graph_refresh_btn = gr.Button("刷新图谱", variant="primary")
            graph_output = gr.Markdown()
            graph_refresh_btn.click(render_graph, outputs=[graph_output])

        with gr.Tab("📈 执行追踪"):
            gr.Markdown("### 执行追踪记录")
            gr.Markdown("实时查看每次问答的节点级追踪数据：耗时、Token 消耗、状态。")

            def get_trace_ui():
                try:
                    from tracer import get_recent_traces, format_trace_html
                    from tracer import get_trace_summary, format_trace_summary_html
                    traces = get_recent_traces(15)
                    summary = get_trace_summary()
                    summary_html = format_trace_summary_html(summary)
                    trace_html = format_trace_html(traces)
                    return summary_html + trace_html
                except Exception as e:
                    return f"Failed to load traces: {e}"

            trace_refresh_btn = gr.Button("刷新追踪", variant="primary")
            trace_output = gr.HTML()
            trace_refresh_btn.click(get_trace_ui, outputs=[trace_output])

    return demo
