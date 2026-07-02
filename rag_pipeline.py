"""RAG 管线 — PDF 加载、分块、向量化、检索"""
import os
import requests
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# === 配置 ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "langgraph_rag_docs"

# DashScope Embedding (OpenAI 兼容模式)
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "text-embedding-v3")

# DashScope LLM (OpenAI 兼容模式)
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL_ID", "qwen-max")


# ═══════════════════════════════════════════
# 自定义 DashScope Embedding（绕过 langchain-openai 格式不兼容）
# ═══════════════════════════════════════════

class DashScopeEmbeddings(Embeddings):
    """DashScope Embedding — 直接调用 REST API，兼容 OpenAI 格式"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码文档"""
        all_vecs = []
        batch_size = 10  # DashScope v3/v4 限制
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_vecs.extend(self._embed_batch(batch))
        return all_vecs

    def embed_query(self, text: str) -> List[float]:
        """编码查询文本"""
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error: {resp.status_code} {resp.text}")
        data = resp.json()
        items = data.get("data") or []
        return [item["embedding"] for item in sorted(items, key=lambda x: x["index"])]


def _get_embeddings() -> DashScopeEmbeddings:
    """获取 DashScope Embedding 实例"""
    return DashScopeEmbeddings(
        model=EMBED_MODEL,
        api_key=EMBED_API_KEY,
        base_url=EMBED_BASE_URL,
    )


def _get_qdrant_client() -> QdrantClient:
    """获取 Qdrant 客户端"""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _ensure_collection(dimension: int = 1024) -> None:
    """确保向量集合存在，并创建 doc_hash 索引"""
    client = _get_qdrant_client()
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        collections = []

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        print(f"[RAG] 创建 Qdrant 集合: {COLLECTION_NAME} (dim={dimension})")
    else:
        print(f"[RAG] 使用现有 Qdrant 集合: {COLLECTION_NAME}")

    # 确保 metadata.doc_hash 字段有索引（用于去重过滤）
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.doc_hash",
            field_schema="keyword",
        )
    except Exception:
        pass  # 索引已存在


def _compute_file_hash(file_path: str) -> str:
    """计算文件的 SHA-256 哈希"""
    import hashlib
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _check_document_indexed(doc_hash: str, doc_name: str) -> bool:
    """检查文档是否已索引"""
    try:
        client = _get_qdrant_client()
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={"must": [{"key": "metadata.doc_hash", "match": {"value": doc_hash}}]},
            limit=1,
        )
        return len(records) > 0
    except Exception:
        return False


def load_and_index_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """加载 PDF 文档，分块并索引到 Qdrant。
    默认启用文件哈希去重：同一文件只索引一次。

    Args:
        pdf_path: PDF 文件路径
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        force_reload: 强制重新索引（忽略去重缓存）

    Returns:
        {"success": bool, "chunks": int, "document": str, "message": str, "cached": bool}
    """
    import uuid
    import time

    if not os.path.exists(pdf_path):
        return {"success": False, "chunks": 0, "document": "", "message": f"文件不存在: {pdf_path}"}

    doc_name = os.path.basename(pdf_path)

    # === 去重检查 ===
    doc_hash = _compute_file_hash(pdf_path)
    if not force_reload and _check_document_indexed(doc_hash, doc_name):
        # 已索引，快速返回
        print(f"[RAG] Cached: {doc_name} (hash={doc_hash[:12]}...)")
        return {
            "success": True,
            "chunks": 0,  # 未重新索引
            "document": doc_name,
            "message": f"[Cached] Document already indexed, skipped ({doc_name})",
            "cached": True,
        }

    try:
        # 1. 加载 PDF
        print(f"[RAG] 正在加载 PDF: {pdf_path}...", flush=True)
        t0 = time.time()
        loader = PyPDFLoader(pdf_path)
        docs: List[Document] = loader.load()
        if not docs:
            return {"success": False, "chunks": 0, "document": "", "message": "PDF 内容为空"}
        print(f"[RAG] PDF 加载完成: {len(docs)} 页, {time.time()-t0:.1f}s", flush=True)

        # 2. 分块
        t0 = time.time()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"[RAG] 分块完成: {len(chunks)} 块, {time.time()-t0:.1f}s", flush=True)

        # 3. 嵌入 + 手动存入 Qdrant
        t0 = time.time()
        embeddings = _get_embeddings()
        test_vec = embeddings.embed_query("test")
        dimension = len(test_vec)
        _ensure_collection(dimension)

        client = _get_qdrant_client()
        texts = [chunk.page_content for chunk in chunks]

        batch_size = 10
        total_stored = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vecs = embeddings.embed_documents(batch_texts)

            points = []
            for j, vec in enumerate(batch_vecs):
                chunk_idx = i + j
                point_id = str(uuid.uuid4())
                points.append({
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "page_content": batch_texts[j],   # QdrantVectorStore 标准字段名
                        "metadata": {
                            "source": doc_name,
                            "doc_hash": doc_hash,
                            "chunk_index": chunk_idx,
                            "page": chunks[chunk_idx].metadata.get("page", -1),
                        },
                    },
                })

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_stored += len(points)
            if (i // batch_size) % 5 == 0:
                print(f"[RAG] 嵌入进度: {min(i+batch_size, len(texts))}/{len(texts)}")

        elapsed = time.time() - t0
        print(f"[RAG] 文档加载完成: {doc_name} → {total_stored} 个分块, {elapsed:.1f}s")
        return {
            "success": True,
            "chunks": total_stored,
            "document": doc_name,
            "message": f"加载成功: {doc_name} ({total_stored} 个分块, {elapsed:.0f}s)",
            "cached": False,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "chunks": 0, "document": "", "message": f"加载失败: {str(e)}"}


# ═══════════════════════════════════════════
# 高级检索：MQE + HyDE
# ═══════════════════════════════════════════

def _generate_mqe_queries(query: str, n: int = 3, model_id: str = "") -> List[str]:
    """MQE（多查询扩展）：用 LLM 生成语义等价的多样化查询"""
    try:
        llm = get_llm(model_id) if model_id else get_llm()
        prompt = [
            {"role": "system", "content": "你是检索查询扩展助手。对用户问题，生成语义等价但表述不同的查询。只输出查询，每行一个，不要编号和标点。"},
            {"role": "user", "content": f"原始查询：{query}\n请生成{n}个不同表述的查询："},
        ]
        text = llm.invoke(prompt).content.strip()
        lines = [ln.strip("- 1234567890. ") for ln in text.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln) > 3][:n]
    except Exception:
        return []


def _generate_hyde_doc(query: str) -> str:
    """HyDE（假设文档嵌入）：用 LLM 生成假设性答案，用答案去检索"""
    try:
        llm = get_llm()
        prompt = [
            {"role": "system", "content": "根据用户问题，写一段客观的答案性段落（不要分析过程），保持中等长度，包含关键术语。直接输出段落。"},
            {"role": "user", "content": f"问题：{query}"},
        ]
        return llm.invoke(prompt).content.strip()
    except Exception:
        return ""


def retrieve_advanced(
    query: str,
    k: int = 5,
    enable_mqe: bool = True,
    enable_hyde: bool = True,
    mqe_expansions: int = 3,
) -> List[Dict[str, Any]]:
    """高级检索：MQE 扩展查询 + HyDE 假设文档 → 多路召回 → 去重合并"""
    # 1. 生成扩展查询
    expansions = [query]
    if enable_mqe:
        mqe_queries = _generate_mqe_queries(query, n=mqe_expansions)
        expansions.extend(mqe_queries)
    if enable_hyde:
        hyde_doc = _generate_hyde_doc(query)
        if hyde_doc and len(hyde_doc) > 10:
            expansions.append(hyde_doc)

    # 2. 对每个扩展查询执行检索
    seen_ids = set()
    all_results = []
    per_query = max(k * 2 // max(len(expansions), 1), 2)

    for q in expansions:
        results = retrieve(q, k=per_query)
        for r in results:
            # 用 content 前 80 字符做简单去重
            sig = r["content"][:80]
            if sig not in seen_ids:
                seen_ids.add(sig)
                all_results.append(r)

    # 3. 按分数排序，取 top-k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:k]


def retrieve(
    query: str,
    k: int = 5,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """从 Qdrant 检索与 query 最相关的文档片段

    Args:
        query: 查询文本
        k: 返回结果数
        score_threshold: 最低相似度阈值

    Returns:
        [{"content": str, "score": float, "source": str}, ...]
    """
    embeddings = _get_embeddings()
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    # 使用带分数的搜索
    docs_with_scores = vectorstore.similarity_search_with_score(
        query, k=k, score_threshold=score_threshold
    )

    results = []
    for doc, score in docs_with_scores:
        results.append({
            "content": doc.page_content,
            "score": float(score),
            "source": doc.metadata.get("source", os.path.basename(doc.metadata.get("source", "unknown"))),
        })

    return results


def build_context(retrieval_results: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """将检索结果拼接为上下文文本，超出则智能截断"""
    parts = []
    total = 0
    for i, r in enumerate(retrieval_results):
        content = r.get("content", "").strip()
        if not content:
            continue
        part = f"[片段 {i+1}] {content}"
        if total + len(part) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(part[:remaining] + "...")
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def get_llm(model_id: str = None):
    """获取 LLM 实例 — 兼容旧接口，转发到 model_factory"""
    from model_factory import create_llm
    if model_id is None:
        model_id = LLM_MODEL or "deepseek-chat"
    return create_llm(model_id)
