"""Agent Planner — 任务拆解与步骤规划"""
import json
from rag_pipeline import get_llm

PLANNER_PROMPT = """你是一个 AI Agent 的任务规划器。分析用户问题，拆解为 2-5 个有序步骤。

可用 action_type:
- "retrieve" — 从知识库检索文档内容
- "tool_call" — 调用外部工具（计算器、网络搜索、读笔记、查时间）
- "answer" — 综合所有信息生成最终答案
- "ask_user" — 信息不足时追问用户

可用工具（tool_call 时在 description 中注明工具名）:
- calculator: 数学计算，输入表达式如 '3*15+2'
- web_search: 联网搜索最新信息
- read_notes: 读取已保存的学习笔记
- get_current_time: 获取当前时间

规则：
1. 计算问题 → tool_call:calculator
2. 时效性问题 → tool_call:web_search
3. 知识库能回答的 → retrieve
4. 如果所有信息都集齐了 → answer
5. 严格输出 JSON，不要任何额外文字

输出格式：
{
  "needs_clarification": false,
  "clarification_question": "",
  "plan": [
    {"step_id": 1, "description": "用计算器算 3*15", "action_type": "tool_call", "tool_name": "calculator"},
    {"step_id": 2, "description": "综合回答", "action_type": "answer"}
  ]
}"""


def plan_task(user_input: str, max_steps: int = 5, model_id: str = "") -> dict:
    """将用户问题拆解为执行计划"""
    llm = get_llm(model_id) if model_id else get_llm()
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"用户问题：{user_input}\n请制定执行计划（JSON）："},
    ]

    try:
        response = llm.invoke(messages)
        text = response.content.strip()

        # 提取 JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        plan_data = json.loads(text)

        # 校验并限制步骤数
        steps = plan_data.get("plan", [])
        if len(steps) > max_steps:
            steps = steps[:max_steps]

        # 补全状态字段
        for i, s in enumerate(steps):
            s["step_id"] = s.get("step_id", i + 1)
            s["status"] = "pending"
            if s.get("action_type") not in ("retrieve", "answer", "ask_user", "tool_call"):
                s["action_type"] = "retrieve"

        return {
            "needs_clarification": plan_data.get("needs_clarification", False),
            "clarification_question": plan_data.get("clarification_question", ""),
            "plan": steps,
        }

    except Exception:
        # 降级：单步 answer
        return {
            "needs_clarification": False,
            "clarification_question": "",
            "plan": [{"step_id": 1, "description": "直接回答", "action_type": "answer", "status": "pending"}],
        }
