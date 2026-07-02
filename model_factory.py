"""多模型统一工厂 — 支持 DeepSeek / 智谱GLM / 通义千问 / OpenAI

v2.0: 优先从 agent_config.yaml 读取配置
"""
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 硬编码后备（当 YAML 配置不可用时使用）──
FALLBACK_MODELS = {
    "deepseek-chat": {
        "name": "DeepSeek V3",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "glm-4-flash": {
        "name": "智谱 GLM-4-Flash (免费)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
    },
    "glm-4-plus": {
        "name": "智谱 GLM-4-Plus",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
    },
    "qwen-max": {
        "name": "通义千问 Qwen-Max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "qwen-plus": {
        "name": "通义千问 Qwen-Plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "gpt-4o-mini": {
        "name": "OpenAI GPT-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-4o": {
        "name": "OpenAI GPT-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def get_available_models() -> list[dict]:
    """返回有有效 API Key 的模型列表"""
    try:
        from config_loader import get_config
        return get_config().get_available_models()
    except Exception:
        pass

    # 降级：使用硬编码
    available = []
    generic_key = os.getenv("LLM_API_KEY", "")
    generic_model = os.getenv("LLM_MODEL_ID", "")

    for model_id, info in FALLBACK_MODELS.items():
        dedicated_key = os.getenv(info["api_key_env"], "")
        if dedicated_key and dedicated_key not in (
            "your-deepseek-api-key", "your-zhipu-api-key",
            "your-qwen-api-key", "your-openai-api-key"
        ):
            available.append({"id": model_id, "name": info["name"]})
        elif generic_key and model_id == generic_model:
            available.append({"id": model_id, "name": info["name"]})

    if not available and generic_key:
        available.append({"id": generic_model or "deepseek-chat",
                          "name": f"当前模型 ({generic_model or 'deepseek-chat'})"})

    return available if available else [{"id": "deepseek-chat", "name": "DeepSeek V3 (需配置 Key)"}]


def create_llm(model_id: str = "deepseek-chat", temperature: float = 0.7) -> ChatOpenAI:
    """创建指定模型的 ChatOpenAI 实例

    优先从 agent_config.yaml 读取，降级到硬编码
    """
    # 尝试从 config 读取
    try:
        from config_loader import get_config
        config = get_config()
        info = config.get_model_info(model_id)
        api_key_env = info.get("api_key_env", "")
        base_url = info.get("base_url", "")
        api_key = os.getenv(api_key_env, "") if api_key_env else ""
    except Exception:
        info = FALLBACK_MODELS.get(model_id, FALLBACK_MODELS["deepseek-chat"])
        api_key_env = info["api_key_env"]
        base_url = info["base_url"]
        api_key = os.getenv(api_key_env, "")

    if not api_key:
        api_key = os.getenv("LLM_API_KEY", "")

    # 如果传了通用配置，覆盖
    generic_url = os.getenv("LLM_BASE_URL", "")
    generic_model = os.getenv("LLM_MODEL_ID", "")
    if model_id == generic_model and generic_url:
        base_url = generic_url

    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
