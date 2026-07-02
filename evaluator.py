"""Agent 评估体系 — 检索质量、答案评分、幻觉检测、成本统计、RAGAS

v2.0: 支持 agent_config.yaml 定价 + RAGAS 深度评估
"""
import os
import json
from datetime import datetime
from typing import Optional


def _get_pricing() -> dict:
    """获取模型定价（优先从 config，降级硬编码）"""
    try:
        from config_loader import get_config
        config = get_config()
        registry = config._get("models.registry", {})
        return {
            mid: (info.get("pricing", {}).get("input", 0.0),
                  info.get("pricing", {}).get("output", 0.0))
            for mid, info in registry.items()
        }
    except Exception:
        pass
    return {
        "deepseek-chat":  (1.0, 2.0),
        "glm-4-flash":    (0.0, 0.0),
        "glm-4-plus":     (50.0, 50.0),
        "qwen-max":       (20.0, 60.0),
        "qwen-plus":      (2.0, 8.0),
        "gpt-4o-mini":    (1.1, 4.4),
        "gpt-4o":         (17.5, 70.0),
    }


def _get_eval_log_path() -> str:
    try:
        from config_loader import get_config
        return get_config().eval_log_path
    except Exception:
        return "eval_log.json"


def _get_log_retention() -> int:
    try:
        from config_loader import get_config
        return get_config().eval_log_retention
    except Exception:
        return 200


EVAL_LOG = os.path.join(os.path.dirname(__file__), _get_eval_log_path())


def load_log() -> list:
    if os.path.exists(EVAL_LOG):
        try:
            with open(EVAL_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_log(entries: list):
    retention = _get_log_retention()
    with open(EVAL_LOG, "w", encoding="utf-8") as f:
        json.dump(entries[-retention:], f, ensure_ascii=False, indent=2)


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """估算单次调用成本"""
    pricing = _get_pricing()
    prices = pricing.get(model_id, (0, 0))
    cost = (input_tokens / 1_000_000) * prices[0] + (output_tokens / 1_000_000) * prices[1]
    return round(cost, 6)


def retrieval_hit_rate(retrieval_docs: list[str], answer: str) -> float:
    """简单检索命中率：答案中出现的检索文档内容比例"""
    if not retrieval_docs or not answer:
        return 0.0
    hits = 0
    for doc in retrieval_docs:
        sentences = [s.strip() for s in doc.replace("\n", "。").split("。") if len(s.strip()) > 10]
        for s in sentences[:5]:
            if s[:30] in answer:
                hits += 1
                break
    return min(hits / max(len(retrieval_docs), 1), 1.0)


def hallucination_check(answer: str, context: str) -> dict:
    """检测幻觉：检查答案中的关键声明是否在上下文中能找到支撑"""
    import re
    if not answer or not context:
        return {"hallucination_rate": 0.0, "details": []}

    sentences = [s.strip() for s in answer.replace("\n", "。").split("。") if len(s.strip()) > 15]
    ungrounded = []

    for s in sentences:
        if re.search(r'\d+', s) and len(s) > 10:
            nums = re.findall(r'\d+[\.\d]*%?', s)
            found = any(n in context for n in nums)
            if not found and len(nums) > 0:
                ungrounded.append(s)

    rate = len(ungrounded) / max(len(sentences), 1)
    return {"hallucination_rate": round(rate, 3), "ungrounded_count": len(ungrounded)}


def create_eval_entry(
    user_input: str,
    answer: str,
    context: str,
    retrieval_docs: list,
    model_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    elapsed_ms: int = 0,
    reflection_score: int = 0,
    success: bool = True,
    ragas_result: Optional[dict] = None,
) -> dict:
    """创建一条完整的评估记录（v2.0 增加 RAGAS 字段）"""
    cost = estimate_cost(model_id, input_tokens, output_tokens)
    hit_rate = retrieval_hit_rate(retrieval_docs, answer)
    hallu = hallucination_check(answer, context)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_id,
        "question": user_input[:120],
        "answer_preview": answer[:150],
        "success": success,
        "reflection_score": reflection_score,
        "retrieval_hit_rate": round(hit_rate, 3),
        "hallucination_rate": hallu["hallucination_rate"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_rmb": cost,
        "elapsed_ms": elapsed_ms,
        "context_length": len(context),
    }

    # RAGAS 深度评估数据
    if ragas_result:
        entry["ragas"] = {
            "overall_score": ragas_result.get("overall_score"),
            "verdict": ragas_result.get("verdict"),
            "metrics": {
                k: {"score": v.get("score"), "reason": v.get("reason", "")[:100]}
                for k, v in ragas_result.items()
                if isinstance(v, dict) and "score" in v
            },
        }

    return entry


def append_eval(entry: dict):
    """追加评估记录到持久化日志"""
    log = load_log()
    log.append(entry)
    save_log(log)


def get_summary() -> dict:
    """获取评估汇总统计（v2.0 增加 RAGAS 统计）"""
    log = load_log()
    if not log:
        return {
            "total": 0, "avg_score": 0, "avg_hit_rate": 0,
            "total_cost": 0, "success_rate": 0, "avg_ragas": 0,
        }

    total = len(log)
    successes = sum(1 for e in log if e.get("success"))
    scores = [e.get("reflection_score", 0) for e in log if e.get("reflection_score")]
    hit_rates = [e.get("retrieval_hit_rate", 0) for e in log]
    costs = [e.get("cost_rmb", 0) for e in log]
    tokens = [e.get("total_tokens", 0) for e in log]

    # RAGAS 统计
    ragas_scores = [
        e.get("ragas", {}).get("overall_score", 0)
        for e in log if e.get("ragas", {}).get("overall_score") is not None
    ]

    return {
        "total": total,
        "success_rate": round(successes / max(total, 1) * 100, 1),
        "avg_score": round(sum(scores) / max(len(scores), 1), 1),
        "avg_hit_rate": round(sum(hit_rates) / max(len(hit_rates), 1), 3),
        "total_cost": round(sum(costs), 4),
        "total_tokens": sum(tokens),
        "models_used": list(set(e.get("model", "?") for e in log)),
        "avg_ragas": round(sum(ragas_scores) / max(len(ragas_scores), 1), 3),
        "ragas_count": len(ragas_scores),
    }


def get_cost_alert() -> Optional[str]:
    """检查成本是否超过预算阈值"""
    try:
        from config_loader import get_config
        config = get_config()
        if not config.cost_tracking_enabled:
            return None
        budget = config.budget_alert_rmb
    except Exception:
        return None

    summary = get_summary()
    if summary["total_cost"] > budget:
        return f"⚠️ 累计成本 ¥{summary['total_cost']:.4f} 已超出预算 ¥{budget:.2f}"
    return None
