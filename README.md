# 智能文档问答助手 — LangGraph Multi-Agent RAG

基于 **LangGraph** 构建的多智能体文档问答系统，实现 Planning / Tool Use / Multi-Agent / Evaluation 四大 Agent 核心能力。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)
![Gradio](https://img.shields.io/badge/Gradio-6.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ 特性

- 🤖 **5-Agent 协作架构** — Planner / Executor / Retrieval / Generator / Evaluator
- 🔍 **高级 RAG 检索** — MQE 多查询扩展 + HyDE 假设文档嵌入
- 📊 **RAGAS 四维评估** — Faithfulness / Answer Relevancy / Context Precision / Context Recall
- 🔧 **工具调用** — 计算器、联网搜索、笔记管理、时间查询
- 🌊 **流式输出** — Agent 协作过程实时可视化
- 🔌 **多模型热切换** — 7 个 LLM 供应商统一工厂
- ⚙️ **配置驱动** — 200+ 参数 YAML 集中管理

## 🏗️ 架构

```
START → record_input → classify_intent
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   load_doc            qa (Agent Loop)      recall/notes/report
        │                  │
        │            planner_node
        │                  │
        │            executor_node
        │                  │
        │            retrieval_node
        │                  │
        │            generator_node
        │                  │
        │            evaluator_node
        │              │    │
        │           retry   │ (ok)
        │              │    │
        │         retrieval  END
        │
       END
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Xhj27165/langgraph-doc-qa.git
cd langgraph-doc-qa
```

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate # Linux/Mac
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

`.env` 示例：

```env
# LLM（至少配置一个）
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com

# Embedding
EMBED_MODEL_NAME=embedding-2
EMBED_API_KEY=your-zhipu-key
EMBED_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 向量数据库
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key

# 可选：联网搜索
TAVILY_API_KEY=tvly-your-key
```

### 4. 启动

```bash
python main.py
```

打开浏览器访问 `http://localhost:7860`

## 📁 项目结构

```
.
├── main.py                  # 启动入口
├── graph.py                 # 图构建
├── state.py                 # DocQAState 状态定义
├── agent_config.yaml        # 全局配置
├── config_loader.py         # 配置加载
├── model_factory.py         # 多模型工厂
├── tools.py                 # 工具定义
├── tool_center.py           # 工具中心
├── planner.py               # 任务规划
├── reflector.py             # 质量反思
├── evaluator.py             # 评估体系
├── ragas_eval.py            # RAGAS 评估
├── rag_pipeline.py          # RAG 管线
├── tracer.py                # 执行追踪
├── ui.py                    # Web 界面
├── agents/
│   ├── supervisor.py        # 图拓扑 + 节点
│   ├── schemas.py           # 子 Agent 状态
│   └── *.py                 # 独立子 Agent
├── requirements.txt
├── .env.example
└── README.md
```

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph 1.2 + LangChain 1.3 |
| LLM | DeepSeek / 智谱 GLM-4 / 通义千问 / GPT-4o |
| 向量检索 | Qdrant Cloud + Embedding v3 |
| 检索增强 | MQE + HyDE |
| 评估 | RAGAS 四维 LLM-as-Judge |
| 前端 | Gradio 6.x |
| 持久化 | SQLite + JSON |

## 📝 配置说明

所有可配置参数集中在 `agent_config.yaml`，包括：

- 模型选择、温度、最大 Token
- 分块大小、重叠量、Top-K
- MQE/HyDE 开关
- 工具启禁用
- RAGAS 阈值
- 预算预警

## 📄 License

MIT License

## 📧 联系

如有问题或建议，欢迎提 Issue 或 PR。
