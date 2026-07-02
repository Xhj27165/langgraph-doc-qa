"""LangGraph 智能文档问答系统 — 启动入口"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()


def main():
    try:
        from config_loader import get_config
        config = get_config()
        print(f"[Main] Config loaded: {config.agent_name} v{config.version}")
        port = config.ui_port
    except Exception:
        config = None
        port = 7860

    print("\n" + "=" * 60)
    print("  📚 智能文档问答助手 (LangGraph)")
    print("  StateGraph + Qdrant + Multi-Agent + RAGAS")
    print("=" * 60)

    try:
        from tool_center import init_tool_center
        init_tool_center()
    except Exception as e:
        print(f"[Main] ToolCenter init skipped: {e}")

    print("  正在启动 Web 界面...\n")

    from ui import create_ui
    import gradio as gr

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
