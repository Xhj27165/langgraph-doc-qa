"""配置加载器 — 单例模式，从 agent_config.yaml 加载所有配置

特性：
- 单例模式，首次访问时加载，后续复用
- 支持环境变量覆盖 (${ENV_VAR} 语法)
- 类型安全的 getter 方法
- YAML 不存在时自动降级为硬编码默认值
"""
import os
import re
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

_CONFIG: Optional["AgentConfig"] = None
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "agent_config.yaml")


def _resolve_env(value: Any) -> Any:
    """递归解析字符串中的 ${ENV_VAR} 占位符"""
    if isinstance(value, str):
        # 匹配 ${VAR_NAME} 模式
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        return re.sub(r'\$\{(\w+)\}', replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _env_or(key_env: str, default: Any = None) -> Optional[str]:
    """读取环境变量（优先），key_env 可以是 'VAR_NAME' 或直接是值"""
    if key_env and key_env.endswith("_KEY") or key_env and key_env.endswith("_URL"):
        return os.getenv(key_env, default)
    return default


class AgentConfig:
    """Agent 全局配置（从 YAML 加载）"""

    def __init__(self, yaml_path: str = _CONFIG_PATH):
        self._data: dict = {}
        self._loaded = False
        self._load_error = None

        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                self._data = _resolve_env(raw) if raw else {}
                self._loaded = True
                print(f"[Config] Loaded: {yaml_path}")
            except ImportError:
                self._load_error = "PyYAML 未安装，使用默认配置"
                print(f"[Config] WARN: {self._load_error}")
            except Exception as e:
                self._load_error = str(e)
                print(f"[Config] WARN: load failed ({e}), using defaults")
        else:
            self._load_error = f"配置文件不存在: {yaml_path}"
            print(f"[Config] WARN: {self._load_error}, using defaults")

    # ── 基础信息 ──

    @property
    def agent_name(self) -> str:
        return self._get("agent.name", "智能文档问答助手")

    @property
    def version(self) -> str:
        return self._get("agent.version", "2.0.0")

    # ── 模型配置 ──

    @property
    def default_model(self) -> str:
        return self._get("models.default", "deepseek-chat")

    def get_model_info(self, model_id: str) -> dict:
        """获取指定模型配置"""
        registry = self._get("models.registry", {})
        return registry.get(model_id, {
            "name": model_id,
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "pricing": {"input": 1.0, "output": 2.0},
        })

    def get_available_models(self) -> list[dict]:
        """返回有 API Key 的模型列表"""
        registry = self._get("models.registry", {})
        available = []
        generic_key = os.getenv("LLM_API_KEY", "")
        generic_model = os.getenv("LLM_MODEL_ID", "")

        for model_id, info in registry.items():
            dedicated_key = os.getenv(info.get("api_key_env", ""), "")
            if dedicated_key and dedicated_key not in (
                "your-deepseek-api-key", "your-zhipu-api-key",
                "your-qwen-api-key", "your-openai-api-key"
            ):
                available.append({"id": model_id, "name": info.get("name", model_id)})
            elif generic_key and model_id == generic_model:
                available.append({"id": model_id, "name": info.get("name", model_id)})

        if not available and generic_key:
            available.append({
                "id": generic_model or "deepseek-chat",
                "name": f"当前模型 ({generic_model or 'deepseek-chat'})"
            })

        return available if available else [
            {"id": "deepseek-chat", "name": "DeepSeek V3 (需配置 Key)"}
        ]

    def get_model_pricing(self, model_id: str) -> tuple:
        """返回 (input_price, output_price) 元/1M tokens"""
        info = self.get_model_info(model_id)
        p = info.get("pricing", {})
        return (p.get("input", 0.0), p.get("output", 0.0))

    # ── Embedding 配置 ──

    @property
    def embed_model(self) -> str:
        return self._get("embedding.model", "text-embedding-v3")

    @property
    def embed_base_url(self) -> str:
        return self._get("embedding.base_url",
                         "https://dashscope.aliyuncs.com/compatible-mode/v1")

    @property
    def embed_api_key_env(self) -> str:
        return self._get("embedding.api_key_env", "EMBED_API_KEY")

    @property
    def embed_batch_size(self) -> int:
        return int(self._get("embedding.batch_size", 10))

    @property
    def embed_dimension(self) -> int:
        return int(self._get("embedding.dimension", 1024))

    def get_embed_api_key(self) -> Optional[str]:
        return os.getenv(self.embed_api_key_env)

    # ── Qdrant 配置 ──

    @property
    def qdrant_url(self) -> Optional[str]:
        url_env = self._get("qdrant.url_env", "QDRANT_URL")
        return os.getenv(url_env)

    @property
    def qdrant_api_key(self) -> Optional[str]:
        key_env = self._get("qdrant.api_key_env", "QDRANT_API_KEY")
        return os.getenv(key_env)

    @property
    def qdrant_collection(self) -> str:
        return self._get("qdrant.collection_name", "langgraph_rag_docs")

    # ── 检索配置 ──

    @property
    def chunk_size(self) -> int:
        return int(self._get("retrieval.chunk_size", 1000))

    @property
    def chunk_overlap(self) -> int:
        return int(self._get("retrieval.chunk_overlap", 200))

    @property
    def retrieval_top_k(self) -> int:
        return int(self._get("retrieval.top_k", 5))

    @property
    def retrieval_score_threshold(self) -> float:
        return float(self._get("retrieval.score_threshold", 0.3))

    @property
    def context_max_chars(self) -> int:
        return int(self._get("retrieval.context_max_chars", 3000))

    @property
    def mqe_enabled(self) -> bool:
        return bool(self._get("retrieval.mqe.enabled", True))

    @property
    def mqe_expansions(self) -> int:
        return int(self._get("retrieval.mqe.expansions", 3))

    @property
    def hyde_enabled(self) -> bool:
        return bool(self._get("retrieval.hyde.enabled", True))

    # ── 工具配置 ──

    @property
    def enabled_tools(self) -> list[str]:
        return self._get("tools.enabled",
                         ["calculator", "web_search", "read_notes", "get_current_time"])

    @property
    def tool_categories(self) -> dict:
        return self._get("tools.categories", {
            "compute": ["calculator"],
            "search": ["web_search"],
            "memory": ["read_notes"],
            "system": ["get_current_time"],
        })

    def get_tool_config(self, tool_name: str) -> dict:
        return self._get(f"tools.tool_configs.{tool_name}", {})

    # ── Agent 配置 ──

    @property
    def planner_max_steps(self) -> int:
        return int(self._get("agents.planner.max_steps", 5))

    @property
    def planner_temperature(self) -> float:
        return float(self._get("agents.planner.temperature", 0.3))

    @property
    def generator_streaming(self) -> bool:
        return bool(self._get("agents.generator.streaming", True))

    @property
    def generator_temperature(self) -> float:
        return float(self._get("agents.generator.temperature", 0.7))

    @property
    def generator_citations(self) -> bool:
        return bool(self._get("agents.generator.add_citations", True))

    @property
    def evaluator_min_answer_len(self) -> int:
        return int(self._get("agents.evaluator.min_answer_length", 20))

    @property
    def evaluator_max_retries(self) -> int:
        return int(self._get("agents.evaluator.max_retries", 2))

    @property
    def evaluator_quality_threshold(self) -> int:
        return int(self._get("agents.evaluator.quality_threshold", 3))

    @property
    def supervisor_max_retries(self) -> int:
        return int(self._get("agents.supervisor.max_plan_retries", 2))

    # ── 评估配置 ──

    @property
    def eval_log_retention(self) -> int:
        return int(self._get("evaluation.basic.log_retention", 200))

    @property
    def ragas_enabled(self) -> bool:
        return bool(self._get("evaluation.ragas.enabled", True))

    @property
    def ragas_metrics(self) -> list[str]:
        return self._get("evaluation.ragas.metrics",
                         ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])

    def get_ragas_threshold(self, metric: str) -> float:
        return float(self._get(f"evaluation.ragas.thresholds.{metric}", 0.6))

    @property
    def cost_tracking_enabled(self) -> bool:
        return bool(self._get("evaluation.cost_tracking.enabled", True))

    @property
    def budget_alert_rmb(self) -> float:
        return float(self._get("evaluation.cost_tracking.budget_alert_rmb", 10.0))

    # ── UI 配置 ──

    @property
    def ui_title(self) -> str:
        return self._get("ui.title", "Document Q&A Assistant")

    @property
    def ui_port(self) -> int:
        return int(self._get("ui.port", 7860))

    @property
    def ui_theme_name(self) -> str:
        return self._get("ui.theme_name", "slate-blue")

    @property
    def ui_dark_mode_default(self) -> bool:
        return bool(self._get("ui.dark_mode_default", False))

    # ── 持久化配置 ──

    @property
    def checkpoint_db(self) -> str:
        return self._get("persistence.checkpoint_db", "checkpoints.db")

    @property
    def eval_log_path(self) -> str:
        return self._get("persistence.eval_log", "eval_log.json")

    @property
    def learning_notes_path(self) -> str:
        return self._get("persistence.learning_notes", "learning_notes.json")

    # ── 内部方法 ──

    def _get(self, dot_path: str, default: Any = None) -> Any:
        """通过点分隔路径访问嵌套字典，如 'models.default'"""
        keys = dot_path.split(".")
        node = self._data
        for key in keys:
            if isinstance(node, dict):
                node = node.get(key)
                if node is None:
                    return default
            else:
                return default
        return node if node is not None else default

    def to_dict(self) -> dict:
        """返回完整配置字典（用于调试）"""
        return dict(self._data)


# ═══════════════════════════════════════════
# 单例入口
# ═══════════════════════════════════════════

def get_config(reload: bool = False) -> AgentConfig:
    """获取全局配置单例"""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = AgentConfig()
    return _CONFIG


def get_config_dict() -> dict:
    """获取完整配置字典（快速访问）"""
    return get_config().to_dict()
