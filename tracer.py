"""Execution Tracer — per-node timing, token capture, trace log persistence

Usage:
    from tracer import RequestTrace, save_trace, get_recent_traces, get_trace_summary

    trace = RequestTrace(question="What is AI?")
    trace.add_node("planner_node", input_preview="...", output_preview="...",
                   input_tokens=500, output_tokens=120, duration_ms=340)
    trace.add_node("generator_node", input_preview="...", output_preview="...",
                   input_tokens=800, output_tokens=300, duration_ms=1200)
    save_trace(trace)
"""
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from config_loader import get_config

TRACE_LOG = os.path.join(os.path.dirname(__file__), "trace_log.json")


@dataclass
class NodeTrace:
    """Single node execution record."""
    node_name: str
    start_time: str = ""
    end_time: str = ""
    duration_ms: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_preview: str = ""
    output_preview: str = ""
    status: str = "ok"  # ok | error


@dataclass
class RequestTrace:
    """Full request trace with per-node records."""
    question: str
    model_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    nodes: list[NodeTrace] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: float = 0
    status: str = "ok"

    def add_node(self, node_name: str, input_preview: str = "", output_preview: str = "",
                 input_tokens: int = 0, output_tokens: int = 0, duration_ms: float = 0,
                 status: str = "ok"):
        """Add a node execution record."""
        self.nodes.append(NodeTrace(
            node_name=node_name,
            start_time="",
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_preview=input_preview[:100],
            output_preview=output_preview[:200],
            status=status,
        ))
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_duration_ms += duration_ms

    def mark_complete(self):
        """Finalize trace totals."""
        self.total_duration_ms = sum(n.duration_ms for n in self.nodes)
        self.total_input_tokens = sum(n.input_tokens for n in self.nodes)
        self.total_output_tokens = sum(n.output_tokens for n in self.nodes)
        error_nodes = [n for n in self.nodes if n.status == "error"]
        if error_nodes:
            self.status = "error"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "question": self.question[:150],
            "model": self.model_id,
            "nodes": [
                {
                    "node": n.node_name,
                    "duration_ms": round(n.duration_ms, 1),
                    "input_tokens": n.input_tokens,
                    "output_tokens": n.output_tokens,
                    "input_preview": n.input_preview,
                    "output_preview": n.output_preview,
                    "status": n.status,
                }
                for n in self.nodes
            ],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "status": self.status,
        }


# ═══════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════

def save_trace(trace: RequestTrace):
    """Save a trace to the persistent log."""
    log = _load()
    log.append(trace.to_dict())
    # Keep last 100 traces
    if len(log) > 100:
        log = log[-100:]
    _save(log)


def get_recent_traces(n: int = 20) -> list[dict]:
    """Return the n most recent traces."""
    log = _load()
    return log[-n:]


def get_trace_summary() -> dict:
    """Aggregate trace statistics."""
    log = _load()
    if not log:
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "avg_duration_ms": 0,
            "node_breakdown": {},
        }

    total = len(log)
    total_input = sum(t.get("total_input_tokens", 0) for t in log)
    total_output = sum(t.get("total_output_tokens", 0) for t in log)
    avg_dur = sum(t.get("total_duration_ms", 0) for t in log) / max(total, 1)

    # Node-level breakdown
    node_counts = {}
    node_durations = {}
    for t in log:
        for n in t.get("nodes", []):
            name = n.get("node", "?")
            node_counts[name] = node_counts.get(name, 0) + 1
            node_durations[name] = node_durations.get(name, 0) + n.get("duration_ms", 0)

    # Cost estimate
    from evaluator import estimate_cost
    models_used = set(t.get("model", "") for t in log)
    total_cost = sum(
        estimate_cost(m, total_input, total_output)
        for m in models_used if m
    )

    return {
        "total_requests": total,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_cost": round(total_cost, 4),
        "avg_duration_ms": round(avg_dur, 1),
        "node_breakdown": {
            name: {"count": node_counts.get(name, 0),
                   "avg_ms": round(node_durations.get(name, 0) / max(node_counts.get(name, 0), 1), 1)}
            for name in sorted(node_counts.keys())
        },
    }


def format_trace_html(traces: list[dict]) -> str:
    """Format recent traces as an HTML table."""
    if not traces:
        return "<p style='color:#94a3b8;'>No traces recorded yet. Ask a question in the Q&A tab first.</p>"

    rows = ""
    for t in reversed(traces):
        ts = t.get("timestamp", "")[:19]
        q = t.get("question", "")[:80]
        model = t.get("model", "")
        nodes_n = len(t.get("nodes", []))
        tokens = f"{t.get('total_input_tokens', 0)}/{t.get('total_output_tokens', 0)}"
        dur = f"{t.get('total_duration_ms', 0):.0f}ms"
        status = t.get("status", "ok")
        badge = "#22c55e" if status == "ok" else "#ef4444"

        # Node detail
        node_detail = "<br>".join(
            f"<span style='font-size:11px;color:#64748b;'>{n.get('node','?')}: "
            f"{n.get('input_tokens',0)}+{n.get('output_tokens',0)}t, "
            f"{n.get('duration_ms',0):.0f}ms</span>"
            for n in t.get("nodes", [])
        )

        rows += f"""
        <tr>
            <td style="font-size:11px;color:#94a3b8;">{ts}</td>
            <td>{q}</td>
            <td>{model}</td>
            <td>{nodes_n}</td>
            <td>{tokens}</td>
            <td>{dur}</td>
            <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{badge};margin-right:4px;"></span>{status}</td>
        </tr>
        <tr>
            <td colspan="7" style="padding:4px 14px 8px 14px;">{node_detail}</td>
        </tr>"""

    return f"""
    <table class="stat-table">
        <tr>
            <th>Time</th><th>Question</th><th>Model</th><th>Nodes</th><th>Tokens (in/out)</th><th>Duration</th><th>Status</th>
        </tr>
        {rows}
    </table>"""


def format_trace_summary_html(summary: dict) -> str:
    """Format trace summary as metric cards."""
    if summary.get("total_requests", 0) == 0:
        return ""

    cards = [
        ("Total Requests", str(summary["total_requests"]), ""),
        ("Total Tokens", f"{summary['total_tokens']:,}",
         f"In: {summary['total_input_tokens']:,} / Out: {summary['total_output_tokens']:,}"),
        ("Avg Duration", f"{summary['avg_duration_ms']:.0f}ms", ""),
        ("Total Cost", f"&yen;{summary['total_cost']:.4f}", ""),
    ]

    html = '<div class="metric-grid">'
    for label, value, sub in cards:
        sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ''
        html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>"""
    html += '</div>'

    # Node breakdown
    if summary.get("node_breakdown"):
        html += '<div style="margin-top:20px;"><div class="section-heading">Node Breakdown</div><table class="stat-table"><tr><th>Node</th><th>Executions</th><th>Avg Duration</th></tr>'
        for name, data in summary["node_breakdown"].items():
            html += f"<tr><td>{name}</td><td>{data['count']}</td><td>{data['avg_ms']:.0f}ms</td></tr>"
        html += "</table></div>"

    return html


# ═══════════════════════════════════════════
# Token extraction helper
# ═══════════════════════════════════════════

def extract_token_usage(response) -> tuple:
    """Extract (input_tokens, output_tokens) from an LLM response.

    Args:
        response: AIMessage or similar from llm.invoke()/llm.stream()

    Returns:
        (input_tokens, output_tokens) tuple, or (0, 0) if not available
    """
    try:
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("token_usage", {}) or meta.get("usage", {})
        if usage:
            return (usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                    usage.get("completion_tokens", 0) or usage.get("output_tokens", 0))
    except Exception:
        pass
    return (0, 0)


def extract_stream_token_usage(chunks: list) -> tuple:
    """Extract token usage from the last chunk of a streaming response."""
    if not chunks:
        return (0, 0)
    return extract_token_usage(chunks[-1])


# ═══════════════════════════════════════════
# Internal
# ═══════════════════════════════════════════

def _load() -> list:
    if os.path.exists(TRACE_LOG):
        try:
            with open(TRACE_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save(log: list):
    with open(TRACE_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
