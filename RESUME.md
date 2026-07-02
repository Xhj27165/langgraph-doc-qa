# 智能文档问答助手 — 基于 LangGraph 的多智能体 RAG 系统

> 简历项目描述 | 实习求职用

---

## 一句话概述

基于 **LangGraph** 构建的多智能体文档问答系统，实现 **Planning / Tool Use / Multi-Agent / Evaluation** 四大 Agent 核心能力，支持 PDF 知识库的智能检索与流式问答。

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **Agent 框架** | LangGraph 1.2 + LangChain 1.3 | StateGraph 状态管理、条件路由、SQLite 持久化 |
| **LLM** | DeepSeek-V3 / 智谱 GLM-4 / 通义千问 / GPT-4o | 多模型热切换，OpenAI 兼容 API 统一调用 |
| **向量检索** | Qdrant Cloud + DashScope Embedding v3 | 文档分块、向量化存储、相似度检索 |
| **检索增强** | MQE（多查询扩展）+ HyDE（假设文档嵌入） | 多路召回 + 去重排序，提升检索命中率 |
| **评估体系** | RAGAS 四维评估（LLM-as-Judge） | Faithfulness / Answer Relevancy / Context Precision / Context Recall |
| **工具调用** | Function Calling + ToolCenter 工具中心 | 计算器、联网搜索(Tavily)、笔记管理、时间查询 |
| **前端** | Gradio 6.x | 流式输出 Chatbot、多 Tab 界面 |
| **持久化** | SQLite (checkpoint) + JSON (eval/trace/notes) | 对话状态持久化、评估日志、执行追踪 |
| **语言** | Python 3.12 | 全栈 Python 实现 |

---

## 项目亮点

### 1. 多智能体协作架构（Supervisor Pattern）

实现了类 AutoGPT 的 Supervisor 调度模式，包含 5 个专职 Agent：

```
Supervisor（调度器）
  ├── Planner Agent    — LLM 任务拆解，自动生成 2-5 步执行计划
  ├── Executor Agent   — Function Calling 工具调度
  ├── Retrieval Agent  — MQE + HyDE 多路检索
  ├── Generator Agent  — 流式 RAG 答案生成 + 引用标注
  └── Evaluator Agent  — 质量评估 + RAGAS 四维评分 + 自动重试
```

- **图拓扑**: 11 个节点 + 2 个条件路由 + 重试闭环（质量不合格 → 自动重新检索）
- **状态管理**: TypedDict 定义 20+ 字段的全局 State，Annotated 实现追加/累加语义
- **持久化**: SQLite Checkpoint 保存对话状态，支持断点续聊

### 2. RAGAS 深度评估体系

自研 LLM-as-Judge 评估管线，对每次问答自动运行四维 RAG 质量评估：

| 指标 | 评估内容 | 阈值 |
|------|----------|------|
| Faithfulness（忠实度） | 答案声明是否在上下文中可验证 | ≥ 0.7 |
| Answer Relevancy（相关性） | 答案是否切题 | ≥ 0.7 |
| Context Precision（精确度） | 检索信噪比 | ≥ 0.6 |
| Context Recall（召回率） | 上下文是否覆盖所有必要信息 | ≥ 0.6 |

- 每个指标使用独立的 LLM Judge Prompt 进行结构化评分（JSON Schema 约束输出）
- 质量不达标自动触发重试闭环（最多 2 次）
- 评估结果持久化到 `eval_log.json`，支持成本追踪和预算预警

### 3. 高级 RAG 检索管线

- **MQE（Multi-Query Expansion）**: LLM 生成 3 个语义等价变体查询，多路召回合并去重
- **HyDE（Hypothetical Document Embedding）**: LLM 生成假设性答案，以答案向量检索
- **文档去重**: SHA-256 哈希 + Qdrant payload 索引，避免重复索引
- **批处理优化**: Embedding 批量调用（batch=10），适配 API 限流

### 4. 工程化能力

- **配置驱动**: 200+ 行 `agent_config.yaml` 集中管理所有参数（模型、工具、Agent、评估阈值）
- **工具中心**: ToolCenter 单例模式，支持工具注册/启禁用/调用统计/延迟监控
- **执行追踪**: `tracer.py` 实现节点级耗时和 Token 消耗追踪，持久化到 `trace_log.json`
- **多模型热切换**: 7 个 LLM 供应商统一工厂，UI 下拉框实时切换
- **流式输出**: `stream_mode="custom"` 实现 Agent 协作过程实时可视化

### 5. Prompt 工程

- **Planner Prompt**: 将自然语言问题拆解为 `retrieve → tool_call → answer` 执行计划
- **Reflector Prompt**: LLM 自评答案质量（完整性/准确性/幻觉），决定是否重试
- **RAGAS Judge Prompts**: 4 个独立的结构化评估 Prompt，JSON Schema 约束输出

---

## 项目规模

| 指标 | 数值 |
|------|------|
| Python 文件 | 18 个 |
| 代码行数 | ~3,500 行 |
| Agent 节点 | 11 个 |
| 工具集成 | 4 个（计算器/搜索/笔记/时间） |
| 支持 LLM | 7 个供应商 |
| 评估指标 | 6 个（Reflection + RAGAS 四维 + Hallucination） |
| UI Tab | 8 个（含工作流图谱和执行追踪） |

---

## 个人贡献（简历话术参考）

可根据实际参与情况选择使用：

- 独立设计并实现了基于 **LangGraph StateGraph** 的多智能体协作架构，包含 Planner / Executor / Retrieval / Generator / Evaluator 五个专职 Agent
- 实现 **Supervisor 调度模式**，通过条件路由实现任务拆解 → 工具调用 → 检索 → 生成 → 评估 → 自动重试的完整闭环
- 自研 **RAGAS 四维 LLM-as-Judge 评估管线**，使用结构化 Prompt + JSON Schema 约束输出，实现问答质量的自动化评分
- 集成 **MQE + HyDE 高级检索策略**，提升检索命中率和答案质量
- 搭建 **ToolCenter 工具中心**，支持工具的注册、启禁用、调用统计和延迟监控
- 构建 **配置驱动的工程架构**，200+ 参数集中管理于 YAML，支持 7 个 LLM 供应商热切换
- 实现 **执行追踪系统**，节点级耗时和 Token 消耗监控，支撑成本优化
- 开发 **Gradio 6.x Web 界面**，支持流式输出和实时 Agent 协作过程可视化

---

## 面试可能问到的问题

### 技术深度

**Q: LangGraph 的 StateGraph 和普通 LangChain Chain 有什么区别？**
> StateGraph 是基于图的状态机，每个节点读写共享 State，支持条件路由和循环。Chain 是线性的 DAG。我这个项目用了 StateGraph 的循环能力实现"评估不通过→重新检索→重新生成→重新评估"的重试闭环。

**Q: MQE 和 HyDE 的原理是什么？为什么能提升检索效果？**
> MQE 通过 LLM 生成同一个问题的多种表述，弥补"用户提问方式"和"文档表述方式"之间的语义 Gap。HyDE 则是先让 LLM 生成一个假设性答案，用答案的向量去检索——因为答案的语言风格更接近文档原文，匹配度更高。

**Q: RAGAS 评估的四个指标分别怎么计算的？**
> 全部基于 LLM-as-Judge。Faithfulness 是把答案拆成独立声明，逐一检查是否在上下文中可验证；Answer Relevancy 是判断答案是否切题、有无遗漏；Context Precision 是评估检索结果的信噪比；Context Recall 是判断上下文是否包含了回答所需的全部信息。每个指标都有独立的 Prompt 模板和 JSON Schema 约束。

**Q: 如何处理 LLM 调用失败或返回格式不正确？**
> Planner 和 Reflector 都有 JSON 解析失败的兜底策略（降级为单步 answer 或默认评分"good"）。工具调用做了 try-catch 包裹。评估重试上限为 2 次。

### 系统设计

**Q: 为什么要用 Supervisor 模式而不是直接把所有逻辑写在一个节点里？**
> 职责分离：规划、检索、生成、评估各司其职。评估不通过时可以只重新检索和生成，不需要重新规划。便于单独调试和优化某个环节。

**Q: 如何保证系统的高可用和可扩展？**
> 配置驱动（改 YAML 不用改代码）、模型热切换（API 挂了可以切备用）、工具中心支持运行时启禁用。文档索引做了 SHA-256 去重避免重复向量化。

**Q: Token 消耗如何控制？**
> 通过执行追踪系统监控每次请求的 Token 消耗，RAGAS 评估可按需触发（不是每次必跑）。MQE 扩展查询数量可配置。上下文长度做了 3000 字符截断。

---

> **相关文档**: `DEVELOPMENT.md`（完整架构文档）
