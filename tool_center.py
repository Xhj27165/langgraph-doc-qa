"""工具中心 — 统一工具注册、发现、启禁用、使用追踪

特性：
- 工具注册表（名称 → 工具对象 + 元数据）
- 按分类索引（compute / search / memory / system）
- 启用/禁用（从 agent_config.yaml 读取）
- 使用统计（调用次数、成功率、平均耗时）
- 动态发现（自动扫描 @tool 装饰的函数）
"""
import time
import threading
from typing import Optional, Callable
from langchain_core.tools import BaseTool
from config_loader import get_config

# ── 工具使用统计 ──

class ToolStats:
    """单工具使用统计（线程安全）"""

    def __init__(self):
        self.lock = threading.Lock()
        self.call_count = 0
        self.success_count = 0
        self.total_latency_ms = 0.0

    def record(self, success: bool, latency_ms: float):
        with self.lock:
            self.call_count += 1
            if success:
                self.success_count += 1
            self.total_latency_ms += latency_ms

    @property
    def success_rate(self) -> float:
        with self.lock:
            return self.success_count / max(self.call_count, 1)

    @property
    def avg_latency_ms(self) -> float:
        with self.lock:
            return self.total_latency_ms / max(self.call_count, 1)

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "calls": self.call_count,
                "success": self.success_count,
                "success_rate": round(self.success_rate, 3),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
            }


# ── 工具注册项 ──

class ToolEntry:
    """单个工具的注册项"""

    def __init__(self, tool: BaseTool, category: str = "general",
                 description: str = "", requires_auth: bool = False):
        self.tool = tool
        self.name = tool.name
        self.category = category
        self.description = description or tool.description
        self.requires_auth = requires_auth
        self.enabled = True
        self.stats = ToolStats()

    def invoke(self, input_str: str) -> str:
        """调用工具并记录统计"""
        t0 = time.time()
        success = True
        try:
            result = self.tool.invoke(input_str)
            return str(result)
        except Exception as e:
            success = False
            return f"[Tool Error] {self.name}: {e}"
        finally:
            elapsed = (time.time() - t0) * 1000
            self.stats.record(success=success, latency_ms=elapsed)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description[:100],
            "enabled": self.enabled,
            "requires_auth": self.requires_auth,
        }


# ── 工具中心 ──

class ToolCenter:
    """全局工具管理中心（单例）"""

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}
        self._by_category: dict[str, list[ToolEntry]] = {}

    def register(self, tool: BaseTool, category: str = "general",
                 requires_auth: bool = False) -> "ToolCenter":
        """注册一个工具"""
        entry = ToolEntry(tool, category=category,
                          description=tool.description,
                          requires_auth=requires_auth)
        self._tools[entry.name] = entry
        self._by_category.setdefault(category, []).append(entry)
        return self

    def get(self, name: str) -> Optional[BaseTool]:
        """获取已启用的工具对象"""
        entry = self._tools.get(name)
        if entry and entry.enabled:
            return entry.tool
        return None

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """获取工具注册项（含统计）"""
        return self._tools.get(name)

    def list_enabled(self) -> list[ToolEntry]:
        """列出所有已启用的工具"""
        return [e for e in self._tools.values() if e.enabled]

    def list_by_category(self, category: str) -> list[ToolEntry]:
        """按分类列出已启用的工具"""
        return [e for e in self._by_category.get(category, []) if e.enabled]

    def list_all(self) -> list[ToolEntry]:
        """列出所有工具（含禁用的）"""
        return list(self._tools.values())

    def get_tools_for_llm(self) -> list[BaseTool]:
        """返回 LangChain Function Calling 格式的工具列表"""
        return [e.tool for e in self.list_enabled()]

    def get_stats(self) -> dict:
        """获取所有工具的使用统计"""
        return {name: e.stats.to_dict() for name, e in self._tools.items()}

    def get_categories(self) -> dict:
        """获取分类 → 工具名映射"""
        return {
            cat: [e.name for e in entries]
            for cat, entries in self._by_category.items()
        }

    def apply_config(self):
        """根据 agent_config.yaml 启用/禁用工具"""
        config = get_config()
        enabled_names = set(config.enabled_tools)

        for name, entry in self._tools.items():
            entry.enabled = name in enabled_names

        enabled = [e.name for e in self.list_enabled()]
        print(f"[ToolCenter] Enabled tools ({len(enabled)}): {', '.join(enabled)}")

    def invoke(self, name: str, input_str: str) -> str:
        """调用已注册工具（带统计）"""
        entry = self._tools.get(name)
        if not entry:
            return f"❌ 工具 '{name}' 未注册"
        if not entry.enabled:
            return f"🔒 工具 '{name}' 已禁用"
        return entry.invoke(input_str)


# ── 单例 ──

_center: Optional[ToolCenter] = None


def get_tool_center() -> ToolCenter:
    """获取工具中心单例"""
    global _center
    if _center is None:
        _center = ToolCenter()
    return _center


def init_tool_center() -> ToolCenter:
    """初始化工具中心：注册所有工具 + 应用配置"""
    from tools import ALL_TOOLS

    center = get_tool_center()
    config = get_config()
    categories = config.tool_categories

    # 确定每个工具的分类
    for tool in ALL_TOOLS:
        cat = "general"
        for c, names in categories.items():
            if tool.name in names:
                cat = c
                break
        center.register(tool, category=cat)

    # 应用配置
    center.apply_config()
    return center
