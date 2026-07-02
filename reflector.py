"""Agent Reflector — 答案质量评估与反思"""
import json
from rag_pipeline import get_llm

REFLECTOR_PROMPT = """你是一个 AI Agent 的质量评估器。评估给出的回答是否充分回答了用户问题。

评估标准：
1. 完整性：回答是否涵盖了问题的所有方面？
2. 准确性：回答是否有事实错误或前后矛盾？
3. 幻觉：回答中是否包含文档未提及的编造内容？

严格输出 JSON：
{
  "quality": "good" | "incomplete" | "hallucination" | "off_topic",
  "score": 1-5,
  "issues": ["问题1", "问题2"],
  "should_retry": true/false,
  "retry_suggestion": "如果要重试，建议补充什么方向？如果不重试，写空字符串"
}"""


def reflect(user_input: str, answer: str, context: str, model_id: str = "") -> dict:
    """评估当前回答的质量

    Returns:
        {"quality": str, "score": int, "issues": list, "should_retry": bool, "retry_suggestion": str}
    """
    # 回答太短或明显错误 → 直接标记
    if not answer or len(answer) < 20:
        return {
            "quality": "incomplete",
            "score": 1,
            "issues": ["回答过短"],
            "should_retry": True,
            "retry_suggestion": "检索更多相关内容后重新回答",
        }

    llm = get_llm(model_id) if model_id else get_llm()
    messages = [
        {"role": "system", "content": REFLECTOR_PROMPT},
        {"role": "user", "content": (
            f"用户问题：{user_input}\n\n"
            f"参考上下文：{context[:1500]}\n\n"
            f"待评估回答：{answer[:2000]}\n\n"
            f"请评估（JSON）："
        )},
    ]

    try:
        response = llm.invoke(messages)
        text = response.content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text)
        return {
            "quality": result.get("quality", "good"),
            "score": result.get("score", 3),
            "issues": result.get("issues", []),
            "should_retry": result.get("should_retry", False),
            "retry_suggestion": result.get("retry_suggestion", ""),
        }
    except Exception:
        return {"quality": "good", "score": 3, "issues": [], "should_retry": False, "retry_suggestion": ""}
