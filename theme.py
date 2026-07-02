"""Enterprise Theme — Slate + Blue + Inter font, dark mode, custom CSS

Usage:
    from theme import build_theme, CSS, JS_CODE, HEAD_HTML, NAV_HTML
    theme = build_theme("slate-blue")
    demo.launch(theme=theme, css=CSS, js=JS_CODE, head=HEAD_HTML)
"""
import gradio as gr


def build_theme(name: str = "slate-blue"):
    """Build a professional Gradio theme.

    Args:
        name: 'slate-blue' | 'monochrome' | 'ocean'
    """
    base = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
        font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
    )

    theme = base.set(
        # ── Body ──
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        body_text_color="*neutral_800",
        body_text_color_dark="*neutral_200",
        body_text_size="*text_sm",

        # ── Block / Card ──
        block_background_fill="white",
        block_background_fill_dark="*neutral_900",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_border_color_dark="*neutral_700",
        block_radius="*radius_lg",
        block_shadow="*shadow_drop_sm",

        # ── Button Primary ──
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_background_fill_dark="*primary_500",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",

        # ── Button Secondary ──
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_hover="*neutral_200",
        button_secondary_background_fill_dark="*neutral_800",
        button_secondary_text_color="*neutral_700",
        button_secondary_text_color_dark="*neutral_300",

        # ── Input ──
        input_background_fill="white",
        input_background_fill_dark="*neutral_900",
        input_border_color="*neutral_300",
        input_border_color_dark="*neutral_600",
        input_border_color_focus="*primary_400",
        input_radius="*radius_md",
        input_shadow_focus="0 0 0 3px rgba(59, 130, 246, 0.15)",
        input_shadow_focus_dark="0 0 0 3px rgba(59, 130, 246, 0.25)",

        # ── Layout ──
        layout_gap="*spacing_md",

        # ── Chatbot ──
        chatbot_text_size="*text_sm",
        code_background_fill="*neutral_100",
        code_background_fill_dark="*neutral_800",

        # ── Stat ──
        stat_background_fill="*primary_50",
        stat_background_fill_dark="*primary_900",
    )

    return theme


# ═══════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════

CSS = """
/* ================================================================
   0. CSS Custom Properties (color tokens)
   ================================================================ */
:root {
    --slate-50: #f8fafc;  --slate-100: #f1f5f9; --slate-200: #e2e8f0;
    --slate-300: #cbd5e1; --slate-400: #94a3b8; --slate-500: #64748b;
    --slate-600: #475569; --slate-700: #334155; --slate-800: #1e293b;
    --slate-900: #0f172a; --slate-950: #020617;
    --blue-50: #eff6ff;   --blue-100: #dbeafe;  --blue-200: #bfdbfe;
    --blue-300: #93c5fd;  --blue-400: #60a5fa;  --blue-500: #3b82f6;
    --blue-600: #2563eb;  --blue-700: #1d4ed8;
    --green-400: #4ade80; --green-500: #22c55e;
    --radius: 8px;
    --radius-lg: 12px;
    --transition: 150ms ease;
}

/* ================================================================
   1. Base / Reset
   ================================================================ */
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* ================================================================
   2. Header
   ================================================================ */
#app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    height: 56px;
    background: white;
    border-bottom: 1px solid var(--slate-200);
    position: sticky;
    top: 0;
    z-index: 100;
}
body.dark #app-header {
    background: var(--slate-900);
    border-bottom-color: var(--slate-700);
}
.header-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.header-logo svg {
    width: 22px; height: 22px;
    color: var(--blue-500);
}
.header-name {
    font-weight: 600;
    font-size: 15px;
    color: var(--slate-800);
    letter-spacing: -0.01em;
}
body.dark .header-name { color: var(--slate-200); }
.header-version {
    font-size: 11px;
    font-weight: 500;
    color: var(--slate-500);
    background: var(--slate-100);
    padding: 2px 8px;
    border-radius: 10px;
    letter-spacing: 0.02em;
}
body.dark .header-version {
    color: var(--slate-400);
    background: var(--slate-800);
}
.header-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--slate-500);
    font-weight: 500;
}
body.dark .header-status { color: var(--slate-400); }
.status-indicator {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green-400);
    box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.2);
}
.status-indicator.online { background: var(--green-500); }

/* ================================================================
   3. Sidebar Navigation
   ================================================================ */
#sidebar-col {
    max-width: 240px !important;
    flex-shrink: 0 !important;
}
.nav-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 0;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    margin: 0 8px;
    border-radius: var(--radius);
    border: none;
    background: transparent;
    color: var(--slate-600);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background var(--transition), color var(--transition), border-left var(--transition);
    text-align: left;
    width: calc(100% - 16px);
    font-family: inherit;
    border-left: 3px solid transparent;
}
.nav-item:hover {
    background: var(--slate-100);
    color: var(--slate-800);
}
body.dark .nav-item:hover {
    background: var(--slate-800);
    color: var(--slate-200);
}
.nav-item.nav-active {
    background: var(--blue-50);
    color: var(--blue-600);
    font-weight: 600;
    border-left: 3px solid var(--blue-500);
}
body.dark .nav-item.nav-active {
    background: rgba(59, 130, 246, 0.12);
    color: var(--blue-400);
    border-left-color: var(--blue-400);
}
.nav-item svg {
    width: 18px; height: 18px;
    flex-shrink: 0;
    opacity: 0.7;
    color: currentColor;
}
.nav-item.nav-active svg { opacity: 1; }

.nav-divider {
    height: 1px;
    background: var(--slate-200);
    margin: 4px 14px;
}
body.dark .nav-divider { background: var(--slate-700); }

/* Dark mode toggle switch */
.theme-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    margin: 4px 8px;
    font-size: 12px;
    color: var(--slate-500);
    font-weight: 500;
}
body.dark .theme-toggle { color: var(--slate-400); }
.theme-switch {
    position: relative;
    width: 40px; height: 22px;
    background: var(--slate-300);
    border-radius: 11px;
    cursor: pointer;
    transition: background var(--transition);
    border: none;
    padding: 0;
}
.theme-switch.active {
    background: var(--blue-500);
}
.theme-switch::after {
    content: '';
    position: absolute;
    top: 2px; left: 2px;
    width: 18px; height: 18px;
    background: white;
    border-radius: 50%;
    transition: transform var(--transition);
    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
}
.theme-switch.active::after {
    transform: translateX(18px);
}

/* ================================================================
   4. Metric Cards
   ================================================================ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 16px 0;
}
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .metric-grid { grid-template-columns: 1fr; } }

.metric-card {
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    transition: box-shadow var(--transition), transform var(--transition);
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
}
body.dark .metric-card {
    background: var(--slate-800);
    border-color: var(--slate-700);
}
body.dark .metric-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--slate-500);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}
body.dark .metric-label { color: var(--slate-400); }
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--slate-900);
    line-height: 1.2;
    letter-spacing: -0.02em;
}
body.dark .metric-value { color: var(--slate-100); }
.metric-sub {
    font-size: 12px;
    color: var(--slate-400);
    margin-top: 4px;
}
body.dark .metric-sub { color: var(--slate-500); }

/* ================================================================
   6. Chatbot
   ================================================================ */
#chatbot-container .bubble-wrap {
    max-width: 80%;
}
.message-row.user .bubble {
    background: var(--blue-500) !important;
    color: white !important;
    border-radius: 14px 14px 4px 14px !important;
}
.message-row.bot .bubble {
    background: var(--slate-100) !important;
    color: var(--slate-800) !important;
    border-radius: 14px 14px 14px 4px !important;
}
body.dark .message-row.bot .bubble {
    background: var(--slate-700) !important;
    color: var(--slate-200) !important;
}
.chatbot {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--slate-200) !important;
}
body.dark .chatbot { border-color: var(--slate-700) !important; }

/* ================================================================
   7. Section headings
   ================================================================ */
.section-heading {
    font-size: 18px;
    font-weight: 600;
    color: var(--slate-800);
    margin: 0 0 16px 0;
    letter-spacing: -0.01em;
}
body.dark .section-heading { color: var(--slate-100); }
.section-desc {
    font-size: 13px;
    color: var(--slate-500);
    margin-bottom: 20px;
    line-height: 1.5;
}
body.dark .section-desc { color: var(--slate-400); }

/* ================================================================
   8. Stat tables (Tools, RAGAS reports)
   ================================================================ */
.stat-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
    margin: 12px 0;
}
.stat-table th {
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 11px;
    color: var(--slate-500);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--slate-200);
}
body.dark .stat-table th { color: var(--slate-400); border-bottom-color: var(--slate-700); }
.stat-table td {
    padding: 10px 14px;
    color: var(--slate-700);
    border-bottom: 1px solid var(--slate-100);
}
body.dark .stat-table td { color: var(--slate-300); border-bottom-color: var(--slate-800); }
.stat-table tr:last-child td { border-bottom: none; }
.stat-table tr:hover td { background: var(--slate-50); }
body.dark .stat-table tr:hover td { background: var(--slate-800); }

.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.status-badge.online { background: #dcfce7; color: #15803d; }
.status-badge.offline { background: var(--slate-100); color: var(--slate-500); }
body.dark .status-badge.online { background: rgba(34,197,94,0.15); color: #4ade80; }
body.dark .status-badge.offline { background: var(--slate-800); color: var(--slate-400); }

/* ================================================================
   9. RAGAS Score Bars
   ================================================================ */
.score-bar-bg {
    width: 100%;
    height: 6px;
    background: var(--slate-100);
    border-radius: 3px;
    margin: 6px 0;
    overflow: hidden;
}
body.dark .score-bar-bg { background: var(--slate-700); }
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 400ms ease;
}
.score-bar-fill.high { background: var(--green-500); }
.score-bar-fill.mid { background: #f59e0b; }
.score-bar-fill.low { background: #ef4444; }

/* ================================================================
   10. Footer
   ================================================================ */
#app-footer {
    margin-top: 40px;
    padding: 16px 24px;
    text-align: center;
    font-size: 12px;
    color: var(--slate-400);
    border-top: 1px solid var(--slate-200);
}
body.dark #app-footer { border-top-color: var(--slate-700); color: var(--slate-500); }
#app-footer a { color: var(--slate-500); text-decoration: none; }
#app-footer a:hover { color: var(--blue-500); }

/* ================================================================
   11. Responsive
   ================================================================ */
@media (max-width: 768px) {
    #app-header { padding: 0 16px; }
    .header-version { display: none; }
    .metric-grid { grid-template-columns: 1fr; }
}
"""


# ═══════════════════════════════════════════════════════════
# JavaScript
# ═══════════════════════════════════════════════════════════

JS_CODE = """
function switchAppTab(index) {
    var tabButtons = document.querySelectorAll('#app-tabs .tab-nav button');
    if (tabButtons.length > index) {
        tabButtons[index].click();
    }
    var navItems = document.querySelectorAll('.nav-item');
    for (var i = 0; i < navItems.length; i++) {
        if (i === index) { navItems[i].classList.add('nav-active'); }
        else { navItems[i].classList.remove('nav-active'); }
    }
}

function toggleDarkMode() {
    document.body.classList.toggle('dark');
    var isDark = document.body.classList.contains('dark');
    localStorage.setItem('gradio-theme', isDark ? 'dark' : 'light');
    var toggle = document.getElementById('theme-switch');
    if (toggle) {
        if (isDark) { toggle.classList.add('active'); }
        else { toggle.classList.remove('active'); }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    if (localStorage.getItem('gradio-theme') === 'dark') {
        document.body.classList.add('dark');
        var toggle = document.getElementById('theme-switch');
        if (toggle) toggle.classList.add('active');
    }
    var firstNav = document.querySelector('.nav-item');
    if (firstNav) firstNav.classList.add('nav-active');
});
"""


# ═══════════════════════════════════════════════════════════
# Head HTML (fonts, meta)
# ═══════════════════════════════════════════════════════════

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<meta name="color-scheme" content="light dark">
"""


# ═══════════════════════════════════════════════════════════
# Navigation HTML (sidebar)
# ═══════════════════════════════════════════════════════════

NAV_HTML = """
<div class="nav-list">
    <button class="nav-item" onclick="switchAppTab(0)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </svg>
        Home
    </button>
    <button class="nav-item" onclick="switchAppTab(1)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        Q&A Chat
    </button>
    <button class="nav-item" onclick="switchAppTab(2)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        Notes
    </button>
    <button class="nav-item" onclick="switchAppTab(3)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"></line>
            <line x1="12" y1="20" x2="12" y2="4"></line>
            <line x1="6" y1="20" x2="6" y2="14"></line>
        </svg>
        Statistics
    </button>
    <button class="nav-item" onclick="switchAppTab(4)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M14.31 8l5.74 9.94H3.95L9.69 8a1.52 1.52 0 0 1 2.62 0z"></path>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
        Tools
    </button>
    <button class="nav-item" onclick="switchAppTab(5)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9 11l3 3L22 4"></path>
            <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
        </svg>
        Evaluation
    </button>
</div>

<div class="nav-divider"></div>

<div class="theme-toggle">
    <span>Dark Mode</span>
    <button id="theme-switch" class="theme-switch" onclick="toggleDarkMode()" title="Toggle dark mode"></button>
</div>
"""


# ═══════════════════════════════════════════════════════════
# Header HTML
# ═══════════════════════════════════════════════════════════

def build_header_html(app_name: str, version: str) -> str:
    return f"""
    <div class="header-brand">
        <span class="header-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
            </svg>
        </span>
        <span class="header-name">{app_name}</span>
        <span class="header-version">v{version}</span>
    </div>
    <div class="header-status">
        <span class="status-indicator online"></span>
        System Online
    </div>"""


# ═══════════════════════════════════════════════════════════
# Metric Cards builder
# ═══════════════════════════════════════════════════════════

def build_metric_cards(stats: dict) -> str:
    """Build an HTML metric card grid from a stats dict."""
    cards = [
        ("Total Questions", str(stats.get("questions", 0)), ""),
        ("Documents", str(stats.get("documents", 0)), stats.get("current_doc", "")),
        ("Success Rate", f"{stats.get('success_rate', 0)}%", f"{stats.get('total_evals', 0)} evaluations"),
        ("Avg Quality", f"{stats.get('avg_score', 0)}/5", f"RAGAS: {stats.get('avg_ragas', 0):.3f}" if stats.get('avg_ragas') else ""),
        ("Total Tokens", f"{stats.get('total_tokens', 0):,}", ""),
        ("Cost", f"¥{stats.get('total_cost', 0):.4f}", f"{stats.get('models_used', '')}"),
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
    return html


# ═══════════════════════════════════════════════════════════
# Footer HTML
# ═══════════════════════════════════════════════════════════

FOOTER_HTML = """
<footer id="app-footer">
    Powered by <strong>LangGraph</strong> &middot; Multi-Agent Document Intelligence &middot; &copy; 2026
</footer>"""
