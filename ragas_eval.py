"""RAGAS 评估 — 4 项核心指标 via LLM-as-Judge

指标说明：
- Faithfulness:      答案中的声明是否都能在上下文中找到支撑
- Answer Relevancy:  答案是否切题、是否充分回应了用户问题
- Context Precision: 检索结果中相关文档是否排在前面（信噪比）
- Context Recall:    上下文是否覆盖了回答所需的所有信息

每个指标返回 0~1 分数 + 详细理由
"""
import json
from typing import Optional
from config_loader import get_config


# ═══════════════════════════════════════════
# 各指标 Prompt 模板
# ═══════════════════════════════════════════

FAITHFULNESS_PROMPT = """你的任务：判断【答案】中的每个事实声明是否都能在【上下文】中找到支撑依据。

评估步骤：
1. 将答案拆解为独立的事实声明（一句一个）
2. 对每个声明，检查上下文中是否有直接或间接支撑
3. 统计：有支撑的声明数 / 总声明数

严格输出 JSON（不要任何额外文字）：
{
  "score": 0.0~1.0,
  "total_claims": <声明总数>,
  "supported_claims": <有支撑数>,
  "unsupported": ["无支撑的声明1", "无支撑的声明2"],
  "reason": "一句话总结评估理由"
}"""


ANSWER_RELEVANCY_PROMPT = """你的任务：判断【答案】是否切中【问题】的要害，是否充分回应。

评估维度：
- 答案是否直接回应了问题的核心？
- 是否有偏题、答非所问的内容？
- 是否遗漏了问题中的子问题？

严格输出 JSON（不要任何额外文字）：
{
  "score": 0.0~1.0,
  "is_relevant": true/false,
  "missing_aspects": ["遗漏的方面1"],
  "off_topic_parts": ["偏题的部分1"],
  "reason": "一句话总结"
}"""


CONTEXT_PRECISION_PROMPT = """你的任务：判断【检索结果】是否精确——相关内容是否排在前面，无关内容占比多少。

检索结果按排名顺序给出（越靠前排名越高）。判断：
- 前面几条是否与问题高度相关？
- 是否有不相关的结果混入？
- 如果重新排序，信噪比如何？

严格输出 JSON（不要任何额外文字）：
{
  "score": 0.0~1.0,
  "relevant_count": <相关的文档片段数>,
  "total_count": <总文档片段数>,
  "noise_indices": [<无关片段序号，从1开始>],
  "reason": "一句话总结"
}"""


CONTEXT_RECALL_PROMPT = """你的任务：判断【上下文】是否包含了回答【问题】所需的全部信息。

分析：
- 要完美回答这个问题，需要哪些关键信息？
- 这些关键信息在上下文中是否都能找到？
- 缺少了哪些必要信息？

严格输出 JSON（不要任何额外文字）：
{
  "score": 0.0~1.0,
  "required_info": ["关键信息点1", "关键信息点2"],
  "covered_info": ["已覆盖的信息点1"],
  "missing_info": ["缺失的信息点1"],
  "reason": "一句话总结"
}"""


# ═══════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════

def _call_judge(prompt_template: str, user_content: str, model_id: str = "") -> dict:
    """调用 LLM-as-Judge 获取结构化评分"""
    from rag_pipeline import get_llm

    llm = get_llm(model_id) if model_id else get_llm()
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": user_content},
    ]

    try:
        response = llm.invoke(messages)
        text = response.content.strip()

        # 提取 JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text)
    except Exception as e:
        return {"score": 0.0, "error": f"LLM judge failed: {str(e)[:120]}"}


def evaluate_faithfulness(answer: str, context: str, model_id: str = "") -> dict:
    """Faithfulness：答案事实声明是否忠于上下文

    Returns:
        {"score": 0.0~1.0, "total_claims": int, "supported_claims": int,
         "unsupported": [...], "reason": str}
    """
    if not answer or not context:
        return {"score": 0.0, "total_claims": 0, "supported_claims": 0,
                "unsupported": [], "reason": "答案或上下文为空"}

    user_content = f"【答案】\n{answer[:2000]}\n\n【上下文】\n{context[:2000]}"
    result = _call_judge(FAITHFULNESS_PROMPT, user_content, model_id)
    result.setdefault("score", 0.0)
    result.setdefault("total_claims", 0)
    result.setdefault("supported_claims", 0)
    result.setdefault("unsupported", [])
    result.setdefault("reason", "")
    return result


def evaluate_answer_relevancy(question: str, answer: str, model_id: str = "") -> dict:
    """Answer Relevancy：答案是否切题

    Returns:
        {"score": 0.0~1.0, "is_relevant": bool, "missing_aspects": [...],
         "off_topic_parts": [...], "reason": str}
    """
    if not question or not answer:
        return {"score": 0.0, "is_relevant": False, "missing_aspects": [],
                "off_topic_parts": [], "reason": "问题或答案为空"}

    user_content = f"【问题】\n{question}\n\n【答案】\n{answer[:2000]}"
    result = _call_judge(ANSWER_RELEVANCY_PROMPT, user_content, model_id)
    result.setdefault("score", 0.0)
    result.setdefault("is_relevant", False)
    result.setdefault("missing_aspects", [])
    result.setdefault("off_topic_parts", [])
    result.setdefault("reason", "")
    return result


def evaluate_context_precision(question: str, retrieval_results: list[dict],
                               model_id: str = "") -> dict:
    """Context Precision：检索结果的信噪比

    Returns:
        {"score": 0.0~1.0, "relevant_count": int, "total_count": int,
         "noise_indices": [...], "reason": str}
    """
    if not retrieval_results:
        return {"score": 0.0, "relevant_count": 0, "total_count": 0,
                "noise_indices": [], "reason": "无检索结果"}

    # 构建带编号的检索结果展示
    parts = []
    for i, r in enumerate(retrieval_results[:10], 1):
        content = r.get("content", "")[:300]
        score = r.get("score", 0)
        parts.append(f"[{i}] score={score:.3f}\n{content}")
    docs_text = "\n\n".join(parts)

    user_content = f"【问题】\n{question}\n\n【检索结果（按排名）】\n{docs_text}"
    result = _call_judge(CONTEXT_PRECISION_PROMPT, user_content, model_id)
    result.setdefault("score", 0.0)
    result.setdefault("relevant_count", 0)
    result.setdefault("total_count", len(retrieval_results))
    result.setdefault("noise_indices", [])
    result.setdefault("reason", "")
    return result


def evaluate_context_recall(question: str, answer: str, context: str,
                            model_id: str = "") -> dict:
    """Context Recall：上下文是否包含回答所需的所有信息

    Returns:
        {"score": 0.0~1.0, "required_info": [...], "covered_info": [...],
         "missing_info": [...], "reason": str}
    """
    if not question or not context:
        return {"score": 0.0, "required_info": [], "covered_info": [],
                "missing_info": [], "reason": "问题或上下文为空"}

    user_content = (f"【问题】\n{question}\n\n"
                    f"【上下文】\n{context[:2000]}\n\n"
                    f"【答案（参考，帮助判断需要哪些信息）】\n{answer[:1000]}")
    result = _call_judge(CONTEXT_RECALL_PROMPT, user_content, model_id)
    result.setdefault("score", 0.0)
    result.setdefault("required_info", [])
    result.setdefault("covered_info", [])
    result.setdefault("missing_info", [])
    result.setdefault("reason", "")
    return result


# ═══════════════════════════════════════════
# 综合评估入口
# ═══════════════════════════════════════════

def run_ragas_evaluation(
    question: str,
    answer: str,
    context: str,
    retrieval_results: list[dict],
    model_id: str = "",
    metrics: Optional[list[str]] = None,
) -> dict:
    """运行完整的 RAGAS 评估

    Args:
        question: 用户问题
        answer: 生成的答案
        context: 检索上下文
        retrieval_results: 检索结果列表
        model_id: 用于评估的 LLM
        metrics: 要计算的指标列表，None = 全部

    Returns:
        {
          "faithfulness": {...},
          "answer_relevancy": {...},
          "context_precision": {...},
          "context_recall": {...},
          "overall_score": 0.0~1.0,
          "warnings": [...],
          "verdict": "pass" | "warning" | "fail",
        }
    """
    config = get_config()
    if metrics is None:
        metrics = config.ragas_metrics

    results = {}
    scores = []
    warnings = []

    if "faithfulness" in metrics:
        f = evaluate_faithfulness(answer, context, model_id)
        results["faithfulness"] = f
        scores.append(f["score"])
        threshold = config.get_ragas_threshold("faithfulness")
        if f["score"] < threshold:
            warnings.append(f"Faithfulness={f['score']:.2f} < {threshold}")

    if "answer_relevancy" in metrics:
        ar = evaluate_answer_relevancy(question, answer, model_id)
        results["answer_relevancy"] = ar
        scores.append(ar["score"])
        threshold = config.get_ragas_threshold("answer_relevancy")
        if ar["score"] < threshold:
            warnings.append(f"AnswerRelevancy={ar['score']:.2f} < {threshold}")

    if "context_precision" in metrics:
        cp = evaluate_context_precision(question, retrieval_results, model_id)
        results["context_precision"] = cp
        scores.append(cp["score"])
        threshold = config.get_ragas_threshold("context_precision")
        if cp["score"] < threshold:
            warnings.append(f"ContextPrecision={cp['score']:.2f} < {threshold}")

    if "context_recall" in metrics:
        cr = evaluate_context_recall(question, answer, context, model_id)
        results["context_recall"] = cr
        scores.append(cr["score"])
        threshold = config.get_ragas_threshold("context_recall")
        if cr["score"] < threshold:
            warnings.append(f"ContextRecall={cr['score']:.2f} < {threshold}")

    overall = round(sum(scores) / max(len(scores), 1), 3)
    verdict = "pass" if overall >= 0.7 else ("warning" if overall >= 0.5 else "fail")

    return {
        **results,
        "overall_score": overall,
        "warnings": warnings,
        "verdict": verdict,
        "metrics_computed": len(scores),
    }


def format_ragas_report(ragas_result: dict) -> str:
    """将 RAGAS 结果格式化为可读报告"""
    lines = ["## 📊 RAGAS 深度评估报告\n"]

    # 总览
    overall = ragas_result.get("overall_score", 0)
    verdict = ragas_result.get("verdict", "?")
    verdict_emoji = {"pass": "✅", "warning": "⚠️", "fail": "❌"}.get(verdict, "❓")
    lines.append(f"**综合评分**: {overall:.3f}  {verdict_emoji} `{verdict}`")
    lines.append(f"**计算指标数**: {ragas_result.get('metrics_computed', 0)}\n")

    # 各指标
    for metric, label in [
        ("faithfulness", "🎯 忠实度"),
        ("answer_relevancy", "🎯 答案相关性"),
        ("context_precision", "📊 上下文精确度"),
        ("context_recall", "📊 上下文召回率"),
    ]:
        if metric in ragas_result:
            m = ragas_result[metric]
            score = m.get("score", 0)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"### {label}: {score:.3f}")
            lines.append(f"`[{bar}]`")
            if m.get("reason"):
                lines.append(f"> {m['reason']}")
            # 详细数据
            if metric == "faithfulness":
                lines.append(f"- 声明总数: {m.get('total_claims', '?')}, "
                             f"有支撑: {m.get('supported_claims', '?')}")
                for u in m.get("unsupported", [])[:3]:
                    lines.append(f"  - ⚠️ 无支撑: {u[:80]}...")
            elif metric == "answer_relevancy":
                for a in m.get("missing_aspects", [])[:3]:
                    lines.append(f"  - ❌ 遗漏: {a}")
            elif metric == "context_precision":
                lines.append(f"- 相关/总数: {m.get('relevant_count', '?')}/{m.get('total_count', '?')}")
                if m.get("noise_indices"):
                    lines.append(f"- 噪音序号: {m['noise_indices']}")
            elif metric == "context_recall":
                lines.append(f"- 需覆盖: {len(m.get('required_info', []))} 点, "
                             f"已覆盖: {len(m.get('covered_info', []))} 点")
                for mi in m.get("missing_info", [])[:3]:
                    lines.append(f"  - 🔍 缺失: {mi}")
            lines.append("")

    # 警告
    if ragas_result.get("warnings"):
        lines.append("### ⚠️ 低于阈值")
        for w in ragas_result["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)
