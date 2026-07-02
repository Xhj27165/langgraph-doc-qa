# 开发文档 — 智能文档问答助手

> **版本**: 2.1.0 | **最后更新**: 2026-06-07 | **状态**: 迭代开发中

---

## 目录

1. [系统架构与数据流](#1-系统架构与数据流)
2. [核心模块](#2-核心模块)
   - [2.1 Prompt 编排](#21-prompt-编排)
   - [2.2 工具集](#22-工具集)
   - [2.3 模型工厂](#23-模型工厂)
   - [2.4 RAG 管线](#24-rag-管线)
   - [2.5 评估体系](#25-评估体系)
3. [接口与协议](#3-接口与协议)
4. [部署与环境](#4-部署与环境)

---

## 1. 系统架构与数据流

### 1.1 架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                   Gradio Web UI (ui.py)                         │
│  8 个 Tab: 首页 | 问答 | 笔记 | 统计 | 工具 | RAGAS | 图谱 | 追踪  │
│              流式输出: stream_mode="custom"                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ graph.invoke() / graph.stream()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            LangGraph StateGraph (agents/supervisor.py)           │
│                                                                  │
│  ┌─────────┐   ┌──────────────┐                                 │
│  │record   │──▶│ classify     │──▶ 简单路径 (→ END):            │
│  │_input   │   │ _intent      │    • index_document  加载文档   │
│  └─────────┘   └──────┬───────┘    • recall_memory    回顾学习   │
│                       │            • add_note         保存笔记   │
│                       │ (qa)       • generate_report  生成报告   │
│                       ▼                                         │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌────────┐        │
│  │planner  │──▶│executor │──▶│retrieval │──▶│generator│        │
│  │_node    │   │_node    │   │_node     │   │_node   │        │
│  └─────────┘   └─────────┘   └──────────┘   └────────┘        │
│       任务拆解      工具调用       多路检索       流式生成        │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────┐   (重试: 质量不足且次数 < 2)                      │
│  │evaluator │───▶ retrieval_node ──▶ generator_node             │
│  │_node     │                                                   │
│  └────┬─────┘                                                   │
│       │ (质量合格)                                               │
│       ▼                                                         │
│      END                                                        │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      持久化层                                    │
│  • SQLite (checkpoints.db) — 图状态检查点                        │
│  • Qdrant Cloud (langgraph_rag_docs) — 向量存储                  │
│  • eval_log.json — 评估记录                                     │
│  • trace_log.json — 执行追踪                                    │
│  • learning_notes.json — 学习笔记                               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流：问答请求（完整路径）

```
用户输入 (Gradio Chatbot)
  │
  ▼
record_input（记录输入）
  读取: user_input
  写入: messages (+HumanMessage), 重置 answer/context/plan/reflection
  附:   启动请求计时器
  │
  ▼
classify_intent（意图识别）
  读取: command（优先）, user_input（回退）
  逻辑: 关键词匹配 → query_type
        "加载"/"上传"  → load_doc
        "回顾"/"学过"  → recall
        "笔记"/"记录"  → notes
        "报告"/"统计"  → report
        其他           → qa
  │
  ▼ (qa)
planner_node（任务规划）
  读取: user_input, selected_model
  调用: plan_task() → LLM 拆解为 2-5 步
  写入: plan[{step_id, description, action_type, tool_name, status}]
  │
  ▼
executor_node（工具执行）
  读取: plan, user_input
  逻辑: 遍历 action_type="tool_call" 的步骤
        → ToolCenter.invoke(tool_name, description)
        → 标记 status="done"
  写入: plan（更新步骤状态）
  │
  ▼
retrieval_node（多路检索）
  读取: user_input, plan
  逻辑: 1. 每个 retrieve 步骤: retrieve_advanced(query, k=3)
        2. 合并去重所有结果
        3. 兜底: retrieve_advanced(user_input, k=5)
  调用: MQE（LLM 生成 3 个变体查询）+ Qdrant 相似度搜索
  写入: context, retrieval_results[{content, score, source}]
  │
  ▼
generator_node（答案生成）
  读取: user_input, context, selected_model, retrieval_results
  逻辑: LLM.stream(system_prompt + user_prompt + context)
        → 通过 get_stream_writer() 流式推送到 UI
        → 从响应元数据中提取 token 用量
        → 附加检索来源引用
  写入: answer, messages (+AIMessage), questions_asked (+1)
  │
  ▼
evaluator_node（质量评估）
  读取: user_input, answer, context, retrieval_results
  逻辑: 1. reflect() — LLM 评分 (1-5)
        2. run_ragas_evaluation() — RAGAS 四维评估（如启用）
        3. create_eval_entry() + append_eval()
        4. RequestTrace → save_trace()
  写入: reflection, execution_log (+条目), retry_count (+1 如重试)
  │
  ▼
route_after_eval（评估路由）
  逻辑: quality ∈ {incomplete, hallucination, off_topic}
        且 retry_count < 2 → 重试（跳回 retrieval_node）
        否则 → END
```

### 1.3 数据流：简单路径

**加载文档**：
```
用户上传 PDF → graph.invoke({command:"load_doc", pdf_file_path})
  → record_input → classify_intent → index_document
  → load_and_index_pdf():
      SHA-256 哈希 → 去重检查 → PyPDFLoader 加载 →
      RecursiveCharacterTextSplitter(1000/200) 分块 →
      DashScopeEmbeddings(批量10) → Qdrant 写入
  → END
```

**回顾学习**：
```
用户输入含"回顾" → classify_intent → recall_memory
  → 遍历 messages 提取问答对
  → 格式化最近 10 条为报告
  → END
```

**保存笔记**：
```
用户保存笔记 → graph.invoke({command:"notes", user_input})
  → record_input → classify_intent → add_note
  → 追加 {timestamp, content, document} 到 learning_notes.json
  → END
```

**生成报告**：
```
用户请求统计 → classify_intent → generate_report
  → 读取会话信息 + 学习笔记数 + evaluator.get_summary()
  → 格式化报告（含成本预警）
  → END
```

### 1.4 状态 Schema（DocQAState）

| 字段 | 类型 | 更新方式 | 说明 |
|------|------|----------|------|
| `messages` | `list[BaseMessage]` | 追加 | 完整对话历史 |
| `user_input` | `str` | 覆盖 | 当前用户输入 |
| `current_document` | `Optional[str]` | 覆盖 | 当前已加载文档名 |
| `pdf_file_path` | `Optional[str]` | 覆盖 | PDF 文件绝对路径 |
| `documents_loaded` | `int` | 覆盖 | 已索引文档计数 |
| `total_chunks` | `int` | 覆盖 | 向量库中分块数 |
| `query_type` | `str` | 覆盖 | 路由: qa/load_doc/recall/notes/report |
| `command` | `str` | 覆盖 | 显式命令覆盖（来自 UI 按钮） |
| `selected_model` | `str` | 覆盖 | 当前选择的 LLM 模型 ID |
| `retrieval_results` | `list[dict]` | 覆盖 | 检索到的文档片段及分数 |
| `context` | `str` | 覆盖 | 合并后的上下文文本 |
| `answer` | `str` | 覆盖 | 最终生成的回答 |
| `plan` | `list[dict]` | 覆盖 | 任务计划: [{step_id, description, action_type, tool_name, status}] |
| `current_step` | `int` | 覆盖 | 当前执行步骤索引 |
| `max_plan_retries` | `int` | 覆盖 | 最大重试次数（默认 2） |
| `reflection` | `str` | 覆盖 | 质量标签: good/incomplete/hallucination/off_topic |
| `needs_clarification` | `bool` | 覆盖 | 是否需要追问 |
| `clarification_question` | `str` | 覆盖 | 追问问题文本 |
| `execution_log` | `list[str]` | 追加 | 执行日志 |
| `retry_count` | `int` | 累加 | 累计重试次数 |
| `questions_asked` | `int` | 累加 | 累计提问次数 |
| `concepts_learned` | `int` | 累加 | 累计保存笔记次数 |
| `session_start` | `Optional[str]` | 覆盖 | 会话开始时间戳 |

### 1.5 文件地图

```
chapter8-langgraph/
├── main.py                  # 入口：加载配置 + 初始化工具中心 + 启动 Gradio
├── graph.py                 # 薄封装 → build_supervisor()
├── state.py                 # DocQAState TypedDict 及 initial_state() 工厂
├── agent_config.yaml        # 所有可配置参数
├── config_loader.py         # AgentConfig 单例（类型安全访问 + 环境变量解析）
├── model_factory.py         # 统一 LLM 工厂（7 个模型）
├── tools.py                 # 4 个工具定义（@tool 装饰器）
├── tool_center.py           # 工具注册中心 + 使用统计（单例）
├── planner.py               # 任务拆解（LLM → 结构化计划）
├── reflector.py             # 答案质量评分（LLM 作裁判）
├── evaluator.py             # 评估指标 + 成本追踪 + 日志持久化
├── ragas_eval.py            # RAGAS 四维评估（LLM 作裁判）
├── rag_pipeline.py          # PDF 加载 + 分块 + 向量化 + 检索
├── tracer.py                # 节点级执行追踪 + 追踪日志
├── theme.py                 # UI 主题（当前未启用）
├── ui.py                    # Gradio 6.x Web 界面（8 个 Tab）
├── checkpoints.db           # SQLite 图状态持久化
├── eval_log.json            # 评估记录（保留最近 200 条）
├── trace_log.json           # 执行追踪（保留最近 100 条）
├── learning_notes.json      # 用户学习笔记
├── requirements.txt         # Python 依赖
├── .env                     # 环境变量（API 密钥、URL 等）
└── agents/
    ├── __init__.py
    ├── schemas.py            # 子 Agent TypedDict（独立，主流程未使用）
    ├── supervisor.py         # **核心**：11 个节点 + 2 个条件路由 + SQLite 检查点
    ├── planner_agent.py      # 独立规划器子图（参考）
    ├── retrieval_agent.py    # 独立检索器子图（参考）
    ├── generator_agent.py    # 独立生成器子图（参考）
    └── evaluator_agent.py    # 独立评估器子图（参考）
```

---

## 2. 核心模块

### 2.1 Prompt 编排

#### 2.1.1 规划器 Prompt（`planner.py`）

**用途**: 将用户问题拆解为有序执行计划。

**当前版本**: v1.0（单轮规划）

```
System: 你是一个 AI Agent 的任务规划器。分析用户问题，拆解为 2-5 个有序步骤。

可用 action_type:
- "retrieve"    — 从知识库检索文档内容
- "tool_call"   — 调用外部工具
- "answer"      — 综合所有信息生成最终答案
- "ask_user"    — 信息不足时追问用户

可用工具（tool_call 时注明工具名）:
- calculator:       数学计算
- web_search:       联网搜索最新信息
- read_notes:       读取已保存的学习笔记
- get_current_time: 获取当前时间

规则：
1. 计算问题 → tool_call:calculator
2. 时效性问题 → tool_call:web_search
3. 知识库能回答的 → retrieve
4. 信息集齐了 → answer
5. 严格输出 JSON，不要任何额外文字
```

**输出格式**:
```json
{
  "needs_clarification": false,
  "clarification_question": "",
  "plan": [
    {"step_id": 1, "description": "描述", "action_type": "retrieve|tool_call|answer|ask_user", "tool_name": "calculator|..."},
    {"step_id": 2, "description": "描述", "action_type": "answer"}
  ]
}
```

**兜底策略**: JSON 解析失败 → 降级为单步 `answer`。

**温度参数**: 0.3（可在 `agents.planner.temperature` 配置）

---

#### 2.1.2 生成器 Prompt（`supervisor.py` — `generator_node`）

**用途**: 严格基于检索上下文生成 RAG 答案。

**当前版本**: v1.0

```
System:  你是专业 AI Agent。严格基于上下文回答。不确定则坦诚说明。中文。
User:    【问题】{user_input}

         【上下文】
         {context}

         请回答：
```

**特性**: 流式输出、自动附加检索来源引用。

**温度参数**: 0.7（可在 `agents.generator.temperature` 配置）

---

#### 2.1.3 反思器 Prompt（`reflector.py`）

**用途**: 在返回用户前，由 LLM 作裁判进行质量评估。

**当前版本**: v1.0

```
System:  你是一个 AI Agent 的质量评估器。评估给出的回答是否充分回答了用户问题。

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
  "retry_suggestion": "重试建议，不重试则为空字符串"
}
```

**短路逻辑**: 回答不足 20 字符 → 直接判定 `quality="incomplete"`, `score=1`, `should_retry=true`。

---

#### 2.1.4 RAGAS 评估 Prompt（`ragas_eval.py`）

**用途**: 四个维度的 RAG 质量深度评估，每个维度使用独立的结构化 Prompt。

**忠实度（Faithfulness）** — 答案中的每个事实声明是否都能在上下文中找到依据？
- 拆解答案为独立声明 → 逐条在上下文中验证 → 统计支撑率
- 返回: `{score (0-1), total_claims, supported_claims, unsupported[], reason}`

**答案相关性（Answer Relevancy）** — 答案是否切中问题要害？
- 评估：是否直接回应、有无偏题内容、是否遗漏子问题
- 返回: `{score, is_relevant, missing_aspects[], off_topic_parts[], reason}`

**上下文精确度（Context Precision）** — 检索结果的信噪比如何？
- 检查排名靠前的文档是否与问题高度相关、无关内容占比
- 返回: `{score, relevant_count, total_count, noise_indices[], reason}`

**上下文召回率（Context Recall）** — 上下文是否包含回答所需的全部信息？
- 识别回答所需的关键信息点 → 检查覆盖情况
- 返回: `{score, required_info[], covered_info[], missing_info[], reason}`

**可配置阈值**（`agent_config.yaml`）:

| 指标 | 阈值 |
|------|------|
| faithfulness | 0.7 |
| answer_relevancy | 0.7 |
| context_precision | 0.6 |
| context_recall | 0.6 |

**判定**: `pass` (综合 ≥ 0.7) | `warning` (≥ 0.5) | `fail` (< 0.5)

---

### 2.2 工具集

所有工具在 `tools.py` 中定义，由 `tool_center.py` 统一管理。

#### 2.2.1 工具清单

| 工具 | 分类 | 函数签名 | 说明 |
|------|------|----------|------|
| `calculator` | compute | `calculator(expression: str) -> str` | 安全数学计算，正则提取表达式，沙盒 `eval()` |
| `web_search` | search | `web_search(query: str) -> str` | Tavily 搜索 API，返回前 3 条结果（标题 + 300 字内容） |
| `read_notes` | memory | `read_notes(query: str = "") -> str` | 读取 `learning_notes.json`，支持关键词过滤，返回最近 5 条 |
| `get_current_time` | system | `get_current_time(_: str = "") -> str` | 返回当前日期时间 |

#### 2.2.2 工具中心（`tool_center.py`）

**核心类**: `ToolCenter`（单例，通过 `get_tool_center()` 获取）

**关键 API**:
```python
center = get_tool_center()
center.register(tool, category="search")     # 注册工具
center.get("calculator")                      # 获取已启用的工具对象
center.list_enabled()                         # 列出所有已启用工具
center.list_by_category("compute")            # 按分类筛选
center.invoke(name, input_str)                # 调用工具（自动记录统计）
center.get_stats()                            # {name: {calls, success_rate, avg_latency_ms}}
center.apply_config()                         # 从 agent_config.yaml 同步启禁用状态
```

**统计**: 线程安全的 `ToolStats` 类，记录每个工具的调用次数、成功率、平均延迟。

#### 2.2.3 工具配置（`agent_config.yaml`）

```yaml
tools:
  enabled:
    - calculator
    - web_search
    - read_notes
    - get_current_time
  categories:
    compute: [calculator]
    search: [web_search]
    memory: [read_notes]
    system: [get_current_time]
  tool_configs:
    web_search:
      max_results: 3
      timeout_sec: 10
    read_notes:
      max_notes: 5
```

---

### 2.3 模型工厂（`model_factory.py`）

**用途**: 通过 OpenAI 兼容 API 为所有供应商提供统一的 `ChatOpenAI` 实例化。

#### 2.3.1 支持的模型

| model_id | 供应商 | API 地址 | 密钥环境变量 |
|----------|--------|----------|-------------|
| `deepseek-chat` | DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `glm-4-flash` | 智谱（免费） | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| `glm-4-plus` | 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| `qwen-max` | 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| `qwen-plus` | 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| `gpt-4o-mini` | OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `gpt-4o` | OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` |

#### 2.3.2 关键函数

```python
def create_llm(model_id: str = "deepseek-chat", temperature: float = 0.7) -> ChatOpenAI

def get_available_models() -> list[dict]  # [{id, name}] 仅返回有有效 API Key 的模型
```

#### 2.3.3 价格（元/百万 Token，在 `agent_config.yaml` 中配置）

| model_id | 输入 | 输出 |
|----------|------|------|
| `deepseek-chat` | ¥1.0 | ¥2.0 |
| `glm-4-flash` | ¥0.0 | ¥0.0 |
| `glm-4-plus` | ¥50.0 | ¥50.0 |
| `qwen-max` | ¥20.0 | ¥60.0 |
| `qwen-plus` | ¥2.0 | ¥8.0 |
| `gpt-4o-mini` | ¥1.1 | ¥4.4 |
| `gpt-4o` | ¥17.5 | ¥70.0 |

---

### 2.4 RAG 管线（`rag_pipeline.py`）

#### 2.4.1 文档索引

```
load_and_index_pdf(pdf_path, chunk_size=1000, chunk_overlap=200, force_reload=False)
  → Dict[{success, chunks, document, message, cached}]
```

**流程**:
1. SHA-256 文件哈希计算 → 去重
2. 去重检查: Qdrant scroll 过滤 `metadata.doc_hash`
3. `PyPDFLoader` → 页面文本
4. `RecursiveCharacterTextSplitter` (分隔符: `\n\n`, `\n`, `。`, `.`, ` `, ``) → 分块
5. `DashScopeEmbeddings` (批量=10) → 向量
6. Qdrant 手动写入 (含 metadata: `source`, `doc_hash`, `chunk_index`, `page`)

**缓存**: 同一文件（SHA-256 匹配）→ 即时跳过，返回 `cached: true`。

#### 2.4.2 检索

```python
# 基础检索
retrieve(query, k=5, score_threshold=None) → [{content, score, source}]

# 高级检索 (MQE + HyDE)
retrieve_advanced(query, k=5, enable_mqe=True, enable_hyde=True)
  → [{content, score, source}]
```

**MQE（多查询扩展）**: LLM 生成 3 个语义等价的变体查询，各自检索后合并去重。

**HyDE（假设文档嵌入）**: LLM 生成假设性答案，作为额外查询用于检索。

**去重**: 取内容前 80 个字符作为签名。

**上下文构建器**: `build_context(results, max_chars=2000)` — 以 `[片段 N]` 标签拼接，智能截断。

#### 2.4.3 嵌入模型（`DashScopeEmbeddings`）

自定义 `Embeddings` 实现，直接调用 OpenAI 兼容的 `/embeddings` REST API。

```python
class DashScopeEmbeddings(Embeddings):
    def embed_documents(texts: List[str]) -> List[List[float]]  # 批量，每次 10 条
    def embed_query(text: str) -> List[float]                    # 单条查询
```

**当前运行时配置**: 智谱 `embedding-2`（384 维），虽然类名叫 "DashScope"。`.env` 文件中通过 `EMBED_MODEL_NAME` 和 `EMBED_BASE_URL` 覆盖了 YAML 配置。

---

### 2.5 评估体系

#### 2.5.1 评估管线

```
evaluator_node
  │
  ├── 1. reflect() → LLM 评分 (1-5)
  │       quality: good | incomplete | hallucination | off_topic
  │
  ├── 2. run_ragas_evaluation() → RAGAS 四维评估（如启用）
  │       faithfulness | answer_relevancy | context_precision | context_recall
  │
  ├── 3. create_eval_entry() → 结构化记录
  │       成本估算 + 命中率 + 幻觉检测 + RAGAS 数据
  │
  ├── 4. append_eval() → eval_log.json（保留 200 条）
  │
  └── 5. RequestTrace → save_trace() → trace_log.json（保留 100 条）
```

#### 2.5.2 评估条目结构

```json
{
  "timestamp": "2026-06-07T12:00:00",
  "model": "deepseek-chat",
  "question": "什么是孤独症？",
  "answer_preview": "根据上下文，孤独症...",
  "success": true,
  "reflection_score": 5,
  "retrieval_hit_rate": 0.333,
  "hallucination_rate": 0.0,
  "input_tokens": 200,
  "output_tokens": 200,
  "total_tokens": 400,
  "cost_rmb": 0.0006,
  "elapsed_ms": 22410,
  "context_length": 992,
  "ragas": {
    "overall_score": 1.0,
    "verdict": "pass",
    "metrics": {
      "faithfulness": {"score": 1.0, "reason": "..."},
      "answer_relevancy": {"score": 1.0, "reason": "..."},
      "context_precision": {"score": 1.0, "reason": "..."},
      "context_recall": {"score": 1.0, "reason": "..."}
    }
  }
}
```

#### 2.5.3 成本估算

公式: `(输入 Token / 1,000,000) × 输入单价 + (输出 Token / 1,000,000) × 输出单价`

预算预警: 累计成本超过 ¥10.00 时在 UI 中显示警告。

---

## 3. 接口与协议

### 3.1 图调用接口

#### 3.1.1 输入（`graph.invoke()` / `graph.stream()`）

```python
# 输入字典（DocQAState 子集）
{
    "user_input": str,              # 必填：用户消息
    "selected_model": str,          # 可选：模型 ID（默认 "deepseek-chat"）
    "command": str,                 # 可选：显式路由 ("load_doc" | "notes" | "report")
    "pdf_file_path": str,           # load_doc 时必填：PDF 绝对路径
}

# 配置
{
    "configurable": {
        "thread_id": str            # 会话/线程标识符，用于状态持久化
    }
}
```

#### 3.1.2 同步输出（`graph.invoke()` 返回值）

```python
# 部分 DocQAState（通过 graph.get_state() 获取）
{
    "answer": str,                  # 最终回答或状态信息
    "context": str,                 # 检索到的上下文（仅问答路径）
    "retrieval_results": list[dict],# 原始检索结果（仅问答路径）
    "current_document": str,        # 已加载文档名（仅加载路径）
    "documents_loaded": int,        # 已索引文档数（仅加载路径）
    "total_chunks": int,            # 分块数（仅加载路径）
    "plan": list[dict],             # 任务计划（仅问答路径）
    "reflection": str,              # 质量标签（仅问答路径）
    "messages": list[BaseMessage],  # 完整对话历史
    "questions_asked": int,         # 累计提问次数
    "concepts_learned": int,        # 累计笔记次数
}
```

#### 3.1.3 流式输出（`graph.stream(stream_mode="custom")`）

每个 yield 的 chunk 是节点内 `get_stream_writer()` 推送的字符串：
- Agent 日志: `"🤖 **[AgentName]** 消息内容\n"`
- 生成器 token: 增量文本内容
- 累积即为最终回答

### 3.2 工具接口

每个工具遵循 LangChain `@tool` 协议：

```python
@tool
def tool_name(param: str) -> str:
    """工具描述（供 LLM function calling 使用）。"""
    # 实现
    return result_string
```

**错误约定**: 工具返回以 `⚠️` 或 `❌` 为前缀的错误字符串，不向图抛出异常。

### 3.3 规划器接口

```python
def plan_task(user_input: str, max_steps: int = 5, model_id: str = "") -> dict:
    """
    返回值:
    {
        "needs_clarification": bool,
        "clarification_question": str,
        "plan": [
            {
                "step_id": int,
                "description": str,
                "action_type": "retrieve" | "tool_call" | "answer" | "ask_user",
                "tool_name": str | None,    # 仅 action_type="tool_call" 时有值
                "status": "pending" | "done"
            }
        ]
    }
    """
```

### 3.4 反思器接口

```python
def reflect(user_input: str, answer: str, context: str, model_id: str = "") -> dict:
    """
    返回值:
    {
        "quality": "good" | "incomplete" | "hallucination" | "off_topic",
        "score": int (1-5),
        "issues": [str],
        "should_retry": bool,
        "retry_suggestion": str
    }
    """
```

### 3.5 RAGAS 评估接口

```python
def run_ragas_evaluation(
    question: str,
    answer: str,
    context: str,
    retrieval_results: list[dict],
    model_id: str = "",
    metrics: Optional[list[str]] = None,
) -> dict:
    """
    返回值:
    {
        "faithfulness": {"score": float, "total_claims": int, "supported_claims": int,
                         "unsupported": [str], "reason": str},
        "answer_relevancy": {"score": float, "is_relevant": bool, "missing_aspects": [str],
                             "off_topic_parts": [str], "reason": str},
        "context_precision": {"score": float, "relevant_count": int, "total_count": int,
                              "noise_indices": [int], "reason": str},
        "context_recall": {"score": float, "required_info": [str], "covered_info": [str],
                           "missing_info": [str], "reason": str},
        "overall_score": float (0.0-1.0),
        "warnings": [str],
        "verdict": "pass" | "warning" | "fail",
        "metrics_computed": int
    }
    """
```

### 3.6 评估记录接口

```python
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
    """
    返回评估条目（结构见第 2.5.2 节）。
    持久化: append_eval(entry)
    汇总:   get_summary() -> {total, success_rate, avg_score, avg_hit_rate,
                              total_cost, total_tokens, models_used,
                              avg_ragas, ragas_count}
    """
```

### 3.7 追踪接口

```python
from tracer import RequestTrace, save_trace, get_recent_traces, get_trace_summary

trace = RequestTrace(question="什么是人工智能？", model_id="deepseek-chat")
trace.add_node("planner_node", input_preview="...", output_preview="...", duration_ms=340)
trace.add_node("generator_node", input_tokens=800, output_tokens=300, duration_ms=1200)
save_trace(trace)

traces = get_recent_traces(20)        # -> list[dict]
summary = get_trace_summary()         # -> {total_requests, total_tokens, avg_duration_ms,
                                      #     node_breakdown: {name: {count, avg_ms}}}
```

---

## 4. 部署与环境

### 4.1 环境要求

| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.12.4 | 基于 Anaconda（Windows） |
| pip | 最新 | 推荐使用虚拟环境 |
| 操作系统 | Windows 11 / Linux / macOS | 跨平台 |
| 网络 | 需外网访问 | LLM API + Embedding API + Qdrant Cloud + Tavily |

### 4.2 Python 依赖

**`requirements.txt`**:
```
langchain>=1.0,<2.0
langchain-core>=1.0,<2.0
langgraph>=1.0,<2.0
langchain-openai
langchain-qdrant
langchain-text-splitters
langchain-community
gradio
python-dotenv
PyYAML>=6.0
```

**已验证安装版本**（`.venv`）:

| 包名 | 版本 |
|------|------|
| `gradio` | 6.15.2 |
| `langchain` | 1.3.2 |
| `langchain-core` | 1.4.0 |
| `langgraph` | 1.2.2 |
| `langgraph-checkpoint` | 4.1.1 |
| `langgraph-checkpoint-sqlite` | 3.1.0 |
| `langchain-openai` | 1.2.2 |
| `langchain-qdrant` | 1.1.0 |
| `langchain-community` | 0.4.2 |
| `langchain-text-splitters` | 1.1.2 |
| `qdrant-client` | 1.18.0 |
| `openai` | 2.38.0 |
| `python-dotenv` | 1.2.2 |
| `PyYAML` | 6.0.3 |

### 4.3 部署步骤

**1. 克隆仓库并创建虚拟环境**:
```bash
git clone <repo>
cd hello-agents/code/chapter8-langgraph
python -m venv .venv
.venv\Scripts\activate     # Windows
```

**2. 安装依赖**:
```bash
pip install -r requirements.txt
pip install PyYAML>=6.0
```

**3. 配置 `.env` 文件**:
```bash
# LLM（以 DeepSeek 为例）
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com

# Embedding（以智谱为例）
EMBED_MODEL_NAME=embedding-2
EMBED_API_KEY=your-zhipu-key
EMBED_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 向量存储（Qdrant Cloud）
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key

# 可选：联网搜索
TAVILY_API_KEY=tvly-your-key
```

**4. 验证配置**:
```bash
python -c "from config_loader import get_config; print(get_config().agent_name)"
# 预期输出: 智能文档问答助手
```

**5. 启动**:
```bash
python main.py
# 访问 http://localhost:7860
```

### 4.4 环境变量参考

| 变量 | 必填 | 默认值 | 用途 |
|------|------|--------|------|
| `LLM_MODEL_ID` | 是 | `deepseek-chat` | 主 LLM 模型 |
| `LLM_API_KEY` | 是 | — | LLM 供应商 API 密钥 |
| `LLM_BASE_URL` | 否 | 供应商默认 | LLM API 端点 |
| `DEEPSEEK_API_KEY` | 否 | — | DeepSeek 专用密钥 |
| `ZHIPU_API_KEY` | 否 | — | 智谱 GLM 专用密钥 |
| `DASHSCOPE_API_KEY` | 否 | — | 通义千问专用密钥 |
| `OPENAI_API_KEY` | 否 | — | OpenAI 专用密钥 |
| `EMBED_MODEL_NAME` | 是 | `text-embedding-v3` | 嵌入模型 ID |
| `EMBED_API_KEY` | 是 | — | 嵌入供应商 API 密钥 |
| `EMBED_BASE_URL` | 是 | 供应商默认 | 嵌入 API 端点 |
| `QDRANT_URL` | 是 | — | Qdrant Cloud 集群 URL |
| `QDRANT_API_KEY` | 是 | — | Qdrant Cloud API 密钥 |
| `TAVILY_API_KEY` | 否 | — | Tavily 搜索 API 密钥 |

### 4.5 配置文件（`agent_config.yaml`）

所有运行时参数集中在 `agent_config.yaml` 中管理。主要配置项：

| 配置段 | 可配置项 |
|--------|----------|
| `models` | 7 个 LLM 供应商的 API 地址、密钥环境变量名、价格 |
| `embedding` | 模型名、API 端点、批量大小、向量维度 |
| `qdrant` | URL/API Key 环境变量名、集合名称、距离函数 |
| `retrieval` | 分块大小、重叠量、Top-K、MQE/HyDE 开关 |
| `tools` | 启用工具列表、分类、各工具配置 |
| `agents` | Planner/Generator/Evaluator 的温度参数、最大步数、最大重试次数 |
| `evaluation` | RAGAS 指标选择与阈值、成本追踪、预算预警 |
| `ui` | 端口号、绑定地址、主题名称 |
| `persistence` | 各类 JSON 日志和 SQLite 数据库的文件路径 |

### 4.6 持久化文件

| 文件 | 格式 | 保留策略 | 用途 |
|------|------|----------|------|
| `checkpoints.db` | SQLite + WAL | 无限 | LangGraph 图状态（对话历史、计划、上下文等） |
| `eval_log.json` | JSON 数组 | 200 条 | 评估条目（成本、质量评分、RAGAS 等） |
| `trace_log.json` | JSON 数组 | 100 条 | 节点级执行追踪（耗时、Token 消耗） |
| `learning_notes.json` | JSON 数组 | 无限 | 用户学习笔记 |

### 4.7 Windows 注意事项

- `main.py` 中包含 `sys.stdout.reconfigure(encoding='utf-8')` 以解决 Windows 控制台 GBK 编码问题
- SQLite 连接使用 `check_same_thread=False` 以支持 Gradio 的多线程事件处理
- 虚拟环境路径：`D:\Agents\hello-agents\.venv`

### 4.8 已知问题与改进方向

1. **Token 计数精度**: 流式生成响应的 `response_metadata` 不一定包含 `token_usage`（取决于供应商）。当前评估/追踪记录中的 token 数量为估算值。

2. **嵌入配置不一致**: `agent_config.yaml` 指定 DashScope `text-embedding-v3`（1024 维），但 `.env` 覆盖为智谱 `embedding-2`（384 维）。运行时以 `.env` 为准。建议统一配置来源。

3. **Qdrant 集合名硬编码**: `rag_pipeline.py` 硬编码 `langgraph_rag_docs`，`.env` 中的 `QDRANT_COLLECTION` 变量未被读取。

4. **无 TTL 机制**: Qdrant 集合和 SQLite 检查点会无限增长，无自动清理策略。

5. **无认证机制**: Gradio UI 绑定 `0.0.0.0`，生产环境需添加 `auth=("user", "pass")`。

6. **同步执行**: LangGraph `invoke()` / `stream()` 是同步阻塞的。生产环境需迁移到异步 `ainvoke()` / `astream()`。

7. **子图未集成**: `agents/` 目录下的独立子图（planner_agent 等）已定义但未在主 Supervisor 中作为编译子图使用，当前直接调用底层函数。

---

> **下次更新计划**: 认证方案、异步迁移、自动化测试、CI/CD 配置。
>
> **维护者**: xhj3320 | **仓库**: hello-agents
