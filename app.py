import base64
import json
from pathlib import Path
import re

import anthropic
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from flame.agent import FlameAgent
from flame.charts import series_chart
from flame.config import DEFAULT_MODEL, SAFETY_COLLECTION, SAFETY_DIR, SAFETY_VECTOR_DB_DIR
from flame.data_store import DataStore
from flame.export import safe_filename, to_csv_bytes, to_xlsx_bytes
from flame.retrieval import ReportIndex
from flame.safety import SafetyLimits, format_limit

st.set_page_config(page_title="EFESTO — NASA Microgravity Fire Safety Dashboard", page_icon=str(Path(__file__).parent / "assets" / "navi_icon.png"), layout="wide")

ALL = "All experiments"
# Simboli tipografici/emoji presenti nei testi estratti dai PDF (es. note a pie' di pagina): non mostrati
_SYMBOLS = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


def plain(text) -> str:
    return _SYMBOLS.sub("", str(text)).strip() if text is not None else ""


KIND_LABELS = {"experiment": "Experiment", "model": "Model", "simulation": "Simulation", "unspecified": "Unspecified"}

ESITO_ICON = {
    "conforme": "compliant",
    "non conforme": "non-compliant",
    "fuori zona di rischio": "outside risk zone",
    "in zona di rischio": "in risk zone"
}

HTML_DASHBOARD_FILE = Path(__file__).parent / "dashboard_psi.html"
with open(HTML_DASHBOARD_FILE, "r", encoding="utf-8") as f:
    PSI_DASHBOARD_HTML = f.read()

HTML_MPH_FILE = Path(__file__).parent / "dashboard_mph.html"
with open(HTML_MPH_FILE, "r", encoding="utf-8") as f:
    MPH_DASHBOARD_HTML = f.read()


def html_clean(content: str):
    """Renderizza HTML grezzo rimuovendo ogni indentazione iniziale per evitare blocchi <pre><code> di Markdown."""
    cleaned = re.sub(r"^[ \t]+", "", content.strip(), flags=re.MULTILINE)
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------------------------------------------------- STILE EFESTO & CENTRATURA SITO
html_clean("""
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

<style>
/* 1. RIMOZIONE COMPLETA DELLA SIDEBAR A SINISTRA (nessuna tab/barra laterale) */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
div[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* 2. CENTRATURA PERFETTA DEL SITO */
.stApp {
    background-color: #0a1626;
    background-image:
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='331' height='331'><g fill='%23e2e8f0'><path d='M40 30 l1.2 4.8 4.8 1.2 -4.8 1.2 -1.2 4.8 -1.2 -4.8 -4.8 -1.2 4.8 -1.2z' opacity='.85'/><path d='M250 95 l.9 3.6 3.6 .9 -3.6 .9 -.9 3.6 -.9 -3.6 -3.6 -.9 3.6 -.9z' opacity='.6'/><path d='M150 210 l1.6 6.4 6.4 1.6 -6.4 1.6 -1.6 6.4 -1.6 -6.4 -6.4 -1.6 6.4 -1.6z' opacity='.9'/><path d='M300 280 l.8 3.2 3.2 .8 -3.2 .8 -.8 3.2 -.8 -3.2 -3.2 -.8 3.2 -.8z' opacity='.5'/><circle cx='95' cy='140' r='1' opacity='.55'/><circle cx='210' cy='30' r='.8' opacity='.45'/><circle cx='60' cy='290' r='1.1' opacity='.6'/><circle cx='280' cy='190' r='.7' opacity='.4'/></g></svg>"),
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='48' height='48'><path d='M0 24.5H48M24.5 0V48' stroke='%2394a3b8' stroke-opacity='.07' fill='none'/><path d='M24.5 22.4 l.5 1.6 1.6 .5 -1.6 .5 -.5 1.6 -.5 -1.6 -1.6 -.5 1.6 -.5z' fill='%23cbd5e1' fill-opacity='.45'/></svg>"),
        radial-gradient(ellipse at 50% 0%, rgba(56, 189, 248, 0.07) 0%, rgba(10, 22, 38, 0) 60%);
    background-attachment: fixed;
    color: #e2e8f0;
    font-family: 'Inter', system-ui, sans-serif;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
div[data-testid="stBottomBlockContainer"] {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
div[data-testid="stBottom"] > div {
    background: rgba(10, 22, 38, 0.92);
}

header[data-testid="stHeader"] { background: transparent; }

::selection {
    background: rgba(245, 158, 11, 0.35);
    color: #fef3c7;
}

/* EFESTO Hero Header */
.efesto-hero {
    padding: 2.2rem 2.6rem;
    border-radius: 20px;
    border: 1px solid rgba(245, 158, 11, 0.35);
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.10) 0%, rgba(15, 23, 42, 0.96) 45%, rgba(11, 17, 32, 0.98) 100%);
    box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.7), 0 0 24px -6px rgba(245, 158, 11, 0.2);
}
.efesto-hero { display: flex; align-items: center; gap: 2rem; }
.efesto-hero-text { flex: 1; min-width: 0; }
.navi-logo {
    width: 170px; height: 170px; flex-shrink: 0; border-radius: 50%;
    box-shadow: 0 0 0 1px rgba(203, 213, 225, 0.25), 0 0 36px -4px rgba(203, 213, 225, 0.25);
}
@media (max-width: 760px) {
    .efesto-hero { flex-direction: column; align-items: flex-start; }
    .navi-logo { width: 110px; height: 110px; }
}
.efesto-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f59e0b;
    margin-bottom: 0.75rem;
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 9999px;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.efesto-title-text {
    margin: 0 0 0.3rem 0;
    font-size: 3rem;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(90deg, #f59e0b 0%, #ef4444 32%, #38bdf8 78%, #ffffff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    letter-spacing: -0.02em;
}
.efesto-subtitle-text {
    font-size: 1.25rem;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 0.85rem;
    font-family: 'Inter', sans-serif;
}
.efesto-desc-text {
    color: #cbd5e1;
    font-size: 1.02rem;
    line-height: 1.65;
    max-width: 960px;
    margin-bottom: 1.4rem;
}

/* Dashboard Banner */
.hero {
    padding: 1.2rem 1.6rem; border-radius: 16px; border: 1px solid #1e293b; margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(17, 24, 39, 0.95) 45%, rgba(15, 23, 42, 0.95) 100%);
    box-shadow: 0 4px 24px -4px rgba(0, 0, 0, 0.6), inset 0 1px 0 0 rgba(255, 255, 255, 0.04);
}
.hero-kicker {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #f59e0b; margin-bottom: 0.3rem; display: inline-block;
    padding: 0.2rem 0.65rem; border-radius: 6px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Custom Feature & Step Cards */
.feat-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.3rem;
    height: 100%;
    box-shadow: 0 4px 20px -4px rgba(0,0,0,0.45);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.feat-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}
.feat-card-header {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.6rem;
}
.feat-card-header .feat-title { flex: 1; line-height: 1.35; }
.feat-card-header sub, .feat-card-header sup { font-size: 0.7em; line-height: 0; }
.feat-card-body {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.55;
}

.step-item {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.3rem 1.1rem;
    height: 100%;
}
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 0.78rem;
    padding: 0.15rem 0.65rem;
    border-radius: 6px;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 10px rgba(245, 158, 11, 0.35);
}
.step-title {
    font-weight: 700;
    font-size: 1rem;
    color: #f8fafc;
    margin-bottom: 0.4rem;
}
.step-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
}

div[data-testid="stExpander"] details { border: 1px solid #1e293b; border-radius: 12px; background: #111827; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
div[data-testid="stExpander"] summary { font-family: 'Inter', sans-serif; color: #e2e8f0; font-weight: 600; }
div[data-testid="stExpander"] summary:hover { color: #38bdf8; }
div[data-testid="stChatMessage"] { border: 1px solid #1e293b; border-radius: 14px; background: #111827; padding: 1rem 1.2rem; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
div[data-testid="stChatInput"] > div { border-radius: 14px; border-color: #1e293b; background: #111827; color: #e2e8f0; }
div[data-testid="stChatInput"] > div:focus-within { border-color: #f59e0b; box-shadow: 0 0 0 3px rgba(245,158,11,0.25); }
div[data-testid="stMetric"] { border: 1px solid #1e293b; border-radius: 12px; background: #111827; padding: 0.8rem 1rem; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
div[data-testid="stMetricValue"] { color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
div[data-testid="stMetricLabel"] { color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 0.78rem; }
.card { border: 1px solid #1e293b; border-radius: 12px; background: #111827; padding: 1rem 1.2rem; margin-bottom: 0.8rem; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
.pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.35); background: rgba(56, 189, 248, 0.12); color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; margin: 0 0.3rem 0.3rem 0; }
.page-text { white-space: pre-wrap; font-size: 0.86rem; color: #cbd5e1; max-height: 520px; overflow-y: auto;
             border: 1px solid #1e293b; border-radius: 10px; padding: 0.9rem 1rem; background: #070b14; font-family: 'JetBrains Mono', monospace; }
hr { border-color: #1e293b !important; }
code { color: #38bdf8 !important; background: rgba(56,189,248,0.12) !important; border: 1px solid rgba(56,189,248,0.25) !important; border-radius: 6px; font-family: 'JetBrains Mono', monospace !important; }

/* Tabs di primo livello */
button[data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-size: 0.92rem; font-weight: 600; color: #94a3b8 !important; padding: 0.6rem 1rem !important; }
button[data-baseweb="tab"]:hover { color: #e2e8f0 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #f59e0b !important; border-bottom: 3px solid #f59e0b !important; }

::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #0a1626; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* Primary CTA Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 60%, #b45309 100%) !important;
    border-color: #f59e0b !important;
    color: #0b1120 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.35) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 60%, #d97706 100%) !important;
    border-color: #fbbf24 !important;
    box-shadow: 0 6px 24px rgba(245, 158, 11, 0.55) !important;
    transform: translateY(-1px) !important;
}

#MainMenu, footer { visibility: hidden; }

/* Loading indicator: twinkling star */
[data-testid="stStatusWidgetRunningManIcon"], [data-testid="stStatusWidgetNewYearsIcon"] { display: none !important; }
[data-testid="stStatusWidgetRunningIcon"] { display: inline-flex; align-items: center; justify-content: center; }
[data-testid="stStatusWidgetRunningIcon"]::after {
    content: ""; width: 18px; height: 18px; display: inline-block; background: #e2e8f0;
    clip-path: polygon(50% 0%, 60% 40%, 100% 50%, 60% 60%, 50% 100%, 40% 60%, 0% 50%, 40% 40%);
    animation: efesto-twinkle 1.6s ease-in-out infinite;
}
@keyframes efesto-twinkle { 0%, 100% { transform: scale(0.7) rotate(0deg); opacity: 0.6; } 50% { transform: scale(1.1) rotate(45deg); opacity: 1; } }
</style>
""")


# --------------------------------------------------------------------------- RISORSE & LOADER INIZIALE
@st.cache_resource(show_spinner=False)
def load_store() -> DataStore:
    return DataStore()


@st.cache_resource(show_spinner=False)
def load_index(_store: DataStore) -> ReportIndex:
    return ReportIndex(_store.pages)


@st.cache_resource(show_spinner=False)
def load_safety() -> tuple[SafetyLimits | None, ReportIndex | None, str | None]:
    try:
        limits = SafetyLimits()
        pages = pd.read_parquet(SAFETY_DIR / "safety_pages.parquet")
        return limits, ReportIndex(pages, SAFETY_VECTOR_DB_DIR, collection=SAFETY_COLLECTION), None
    except Exception as e:
        return None, None, str(e)


store = load_store()
index = load_index(store)
safety, safety_index, safety_error = load_safety()

ds_info = store.datasets.set_index("source_file")


def secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


@st.cache_resource
def get_client(api_key: str, workspace_id: str | None) -> anthropic.Anthropic:
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return anthropic.Anthropic(api_key=api_key, default_headers=headers, max_retries=3)


@st.cache_data(show_spinner=False, max_entries=64)
def csv_bytes(df: pd.DataFrame) -> bytes:
    return to_csv_bytes(df)


@st.cache_data(show_spinner=False, max_entries=64)
def xlsx_bytes(df: pd.DataFrame, title: str, meta_json: str) -> bytes:
    return to_xlsx_bytes(df, title, json.loads(meta_json))


def download_buttons(df: pd.DataFrame, title: str, key: str, meta: dict | None = None):
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("CSV", csv_bytes(df), safe_filename(title, "csv"), "text/csv",
                       key=f"{key}_csv", on_click="ignore", width="stretch")
    c2.download_button("Excel", xlsx_bytes(df, title, json.dumps(meta or {}, default=str)), safe_filename(title, "xlsx"),
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"{key}_xlsx", on_click="ignore", width="stretch")


# --------------------------------------------------------------------------- STATO & NAVIGAZIONE LINEARE
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

st.session_state.setdefault("messages", [])
st.session_state.setdefault("api_history", [])


def go_to_dashboard():
    st.session_state.current_page = "dashboard"


def go_to_home():
    st.session_state.current_page = "home"


api_key = secret("ANTHROPIC_API_KEY")
model = secret("ANTHROPIC_MODEL", DEFAULT_MODEL)


# --------------------------------------------------------------------------- RENDERING RISPOSTE ASSISTENTE
TAB_AGENT = "AI Co-pilot"
TAB_DATA = "Experimental Data & Series"
TAB_PSI = "PSI Simulator (g/gₑ)"
TAB_MPH = "MPH Lunar Module (0.165 gₑ)"
TAB_REPORTS = "PSI Scientific Reports"
TAB_SAFETY = "NASA Safety Standards"
ALL_DASHBOARD_TABS = [
    TAB_AGENT,
    TAB_DATA,
    TAB_PSI,
    TAB_MPH,
    TAB_REPORTS,
    TAB_SAFETY,
]


def apply_explore(sugg: dict):
    if st.session_state.get("exp_tag", ALL) not in (ALL, sugg["experiment_tag"]):
        st.session_state.exp_tag = ALL
    st.session_state.explore_target = sugg
    st.session_state.dashboard_active_tab = TAB_DATA


def render_explore(guides: list[dict], key: str):
    if not guides:
        return
    st.markdown("**Charts of this data in the Data & Series tab**")
    for i, g in enumerate(guides):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.markdown(f"**{g['figure'] or g['dataset']}** — {g['y_description']} · {len(g['series_ids'])} series")
        c2.button("View in the Data tab →", key=f"{key}_explore_{i}", on_click=apply_explore, args=(g,),
                  width="stretch", help=f"Dataset {g['dataset']} · quantity {g['y_var']}")


def render_sources(msg: dict):
    hits, files = msg.get("doc_hits", []), msg.get("data_sources", [])
    if not hits and not files:
        return
    with st.expander(f"Sources consulted ({len(hits)} report pages, {len(files)} data files)"):
        if files:
            st.markdown("**Tabular data**")
            for f in files:
                info = ds_info.loc[f] if f in ds_info.index else None
                extra = ""
                if info is not None and pd.notna(info.get("reference_doi")):
                    extra = f" — {info['journal']} {info['year']}, {info['figure']} · [DOI]({info['reference_doi']})"
                st.markdown(f"- `{f}`{extra}")
        if hits:
            st.markdown("**Reports**")
            for h in hits:
                st.markdown(f"- `{h['tag']}` — `{h['file']}` · **p. {h['page']}**" + (f" · _{h['section']}_" if h["section"] else ""))
                st.caption(h["text"][:300].replace("\n", " ") + "…")


def render_assistant(msg: dict, key: str):
    st.markdown(msg["content"])
    render_explore(msg.get("explores", []), key)
    for i, ch in enumerate(msg.get("charts", [])):
        st.plotly_chart(ch["figure"], width="stretch", key=f"{key}_chart_{i}")
    for t in msg.get("tables", []):
        st.markdown(f"**{t['title']}** · {len(t['df'])} rows" + (" · truncated" if t.get("truncated") else ""))
        st.dataframe(t["df"], width="stretch", height=min(420, 38 + 35 * len(t["df"])), hide_index=True)
        download_buttons(t["df"], t["title"], f"{key}_{t['id']}",
                         {"SQL query": t.get("sql") or "original table", "sources": t.get("sources", [])})
    render_sources(msg)
    if msg.get("trace") and secret("DEBUG_MODE", False):
        with st.expander("Analysis details"):
            for step in msg["trace"]:
                st.markdown(f"{'[OK]' if step['ok'] else '[ERROR]'} {step['label']}")
                if step["tool"] == "query_data":
                    st.code(step["input"].get("sql", ""), language="sql")
                if not step["ok"]:
                    st.caption(f"Error (automatically corrected by the model when possible): {step.get('error')}")
            u = msg.get("usage") or {}
            st.caption(f"Tokens — input: {u.get('input_tokens', 0):,} · from cache: {u.get('cache_read_input_tokens', 0):,} · "
                       f"output: {u.get('output_tokens', 0):,}")


def to_message(result) -> dict:
    a = result.artifacts
    return {
        "role": "assistant", "content": result.text, "usage": result.usage,
        "charts": [{"title": c.title, "figure": c.figure} for c in a.charts],
        "tables": [{"id": t.id, "title": t.title, "df": t.df, "sql": t.sql, "truncated": t.truncated, "sources": t.sources}
                   for t in a.displayed_tables],
        "doc_hits": [{"file": f, "page": p, **v} for (f, p), v in sorted(a.doc_hits.items())],
        "data_sources": a.data_sources,
        "trace": a.trace,
        "explores": a.explores,
    }


# --------------------------------------------------------------------------- LOGO NAVI
ASSETS_DIR = Path(__file__).parent / "assets"


@st.cache_data(show_spinner=False)
def logo_data_uri(name: str = "navi_logo.png") -> str:
    path = ASSETS_DIR / name
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


# --------------------------------------------------------------------------- VISUALIZZATORE IMMAGINE NASA
def render_nasa_flame_visual():
    svg_path = Path(__file__).parent / "assets" / "nasa_flex_flame.svg"
    rendered = False
    if svg_path.exists():
        try:
            with open(svg_path, "r", encoding="utf-8") as f:
                svg_content = f.read()
            clean_svg = re.sub(r"^[ \t]+", "", svg_content.strip(), flags=re.MULTILINE)
            clean_html = (
                '<div class="flame-image-container" style="border-radius:16px; overflow:hidden; border:1px solid #1e293b; box-shadow:0 8px 32px rgba(0,0,0,0.6); margin-bottom:0.75rem;">\n'
                + clean_svg
                + '\n</div>'
            )
            html_clean(clean_html)
            rendered = True
        except Exception:
            rendered = False

    if not rendered:
        fallback_html = """
<div style="background:#0f172a; border:1px solid #1e293b; border-radius:16px; padding:2rem; text-align:center; margin-bottom:0.75rem;">
<h3 style="color:#38bdf8; margin:0 0 0.5rem 0;">NASA FLEX-2 • Microgravity Droplet Combustion</h3>
<p style="color:#94a3b8; font-size:0.9rem; max-width:620px; margin:0 auto;">
Isolated droplet combustion in microgravity (g<sub>eff</sub> ≈ 10<sup>-5</sup> g<sub>e</sub>). Without thermal buoyancy the flame takes on a perfectly spherical symmetry dominated by pure diffusive transport and cool-flame regimes.
</p>
</div>
"""
        html_clean(fallback_html)

    st.caption("**NASA FLEX experiment (Flame Extinction Experiment) • ISS Combustion Integrated Rack (CIR)** — "
               "Steady spherical flame structure around an isolated hydrocarbon droplet suspended on quartz fibers. "
               "In pure microgravity (Ra ≈ 0) natural convection is completely suppressed: the flame burns under purely diffusive conditions "
               "and reveals the transition to *cool flames* at ~550 K before final radiative extinction.")


# ===========================================================================
# VISTA 1 — HOME PAGE (SCHERMATA INIZIALE PULITA, 1 SOLO PUNTO D'INGRESSO)
# ===========================================================================
def render_home_page():
    # 1. Hero Section
    html_clean("""
<div class="efesto-hero">
<img class="navi-logo" src="{logo}" alt="NAVI logo">
<div class="efesto-hero-text">
<div class="efesto-badge">EFESTO · NASA PSI · LUNAR HABITAT FIRE SAFETY ENGINEERING</div>
<div class="efesto-title-text">EFESTO</div>
<div class="efesto-subtitle-text">Flame in Freefall — NASA Microgravity Fire Safety Dashboard</div>
<div class="efesto-desc-text">
A multiphysics engineering and advanced artificial intelligence platform for fire safety
in deep space exploration and in the pressurized habitats of the Artemis program.<br>
EFESTO combines open data from <b>NASA Physical Sciences Informatics (PSI)</b> with computational simulations,
droplet combustion models in microgravity (0 g) and partial lunar gravity (0.165 g<sub>e</sub>), rigorous verification
against the <b>NASA-STD-3001</b> and <b>NASA-STD-6001</b> standards, and an autonomous <b>AI Co-pilot</b> with citations verified down to the single page.
</div>
</div>
</div>
""".replace("{logo}", logo_data_uri()))

    # UNICO PUNTO D'INGRESSO NELLA HERO
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2.5, 1])
    with c_btn2:
        st.button("Enter the EFESTO Dashboard", type="primary", on_click=go_to_dashboard, key="hero_enter_cta", width="stretch")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Immagine NASA FLEX + Fisica del fuoco
    c_img, c_info = st.columns([1.15, 0.85], gap="large")
    with c_img:
        st.markdown("#### Droplet Combustion in Microgravity (NASA FLEX)")
        render_nasa_flame_visual()

    with c_info:
        st.markdown("#### The Physics of Fire at g ≈ 0")
        html_clean("""
<div class="feat-card" style="margin-bottom: 0.8rem;">
<div class="feat-card-header" style="color: #38bdf8;">
<span class="feat-title">No Thermal Buoyancy</span>
</div>
<div class="feat-card-body">
On Earth, buoyancy stretches the flame into a rising convective column.
In free fall (g ≈ 10<sup>-5</sup> g<sub>e</sub>), with Rayleigh number Ra ≈ 0,
the plume disappears: the flame takes on a <b>perfect spherical symmetry</b> controlled only by molecular diffusion.
</div>
</div>

<div class="feat-card" style="margin-bottom: 0.8rem;">
<div class="feat-card-header" style="color: #f59e0b;">
<span class="feat-title">Low-Temperature Regime (Cool Flames)</span>
</div>
<div class="feat-card-body">
In hydrocarbon droplets (e.g. n-heptane in the FLEX experiment), after the apparent extinction of the visible high-temperature flame,
the chemical reaction continues in a <b>500–750 K</b> regime that is almost invisible to the naked eye, still consuming fuel.
</div>
</div>

<div class="feat-card">
<div class="feat-card-header" style="color: #ef4444;">
<span class="feat-title">Transition to Lunar Gravity (0.165 g<sub>e</sub>)</span>
</div>
<div class="feat-card-body">
On the Moon, reduced gravity drives a slow but insidious convection. In habitat modules such as the Multi-Purpose Habitat (MPH),
extinction limits and flame spread rates over cables and fabrics differ radically from tests performed on Earth.
</div>
</div>
""")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # 3. Sezione Obiettivi
    st.markdown("### EFESTO Scientific & Engineering Objectives")
    st.caption("Designed for aerospace engineers, microgravity researchers and safety officers of crewed space missions.")

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#38bdf8;">
<span>01</span> <span class="feat-title">Diffusive Combustion</span>
</div>
<div class="feat-card-body">
Characterize chemical kinetics and radiative extinction in the complete absence of natural convection (0 g) to prevent uncontrolled ignitions in orbital laboratories.
</div>
</div>
""")

    with o2:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#f59e0b;">
<span>02</span> <span class="feat-title">PSI Data & Models</span>
</div>
<div class="feat-card-body">
Correlate more than 15,000 experimental points from the NASA PSI repository (PSI-117, BASS, Saffire) with the d<sup>2</sup>-law and analytical models.
</div>
</div>
""")

    with o3:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#10b981;">
<span>03</span> <span class="feat-title">Standard NASA-STD</span>
</div>
<div class="feat-card-body">
Automatically validate simulations and operational scenarios against the binding requirements of NASA-STD-3001 (Human System) and NASA-STD-6001 (Flammability).
</div>
</div>
""")

    with o4:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#a855f7;">
<span>04</span> <span class="feat-title">RAG Co-pilot</span>
</div>
<div class="feat-card-body">
Provide transparent, evidence-based querying with sandboxed SQL, dynamic Plotly charts and page-level citations.
</div>
</div>
""")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # 4. Card delle Funzionalità (schede pulite senza bottoni ridondanti)
    st.markdown("### Platform Features")
    st.caption("A complete suite of analytical, multiphysics and bibliographic tools available inside the Dashboard.")

    f1, f2, f3 = st.columns(3)
    with f1:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#38bdf8;">
<span class="feat-title">AI Co-pilot & Scientific RAG</span>
</div>
<div class="feat-card-body">
Engineering assistant connected to DuckDB and ChromaDB. It queries experimental datasets with safe SQL, builds Plotly charts and cites the exact page of the NASA report.
</div>
</div>
""")

    with f2:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#f59e0b;">
<span class="feat-title">Experimental Data & Series</span>
</div>
<div class="feat-card-body">
Explore 67 historical time series from space missions with multi-series selection, logarithmic axes, aggregate statistics and CSV or Excel export.
</div>
</div>
""")

    with f3:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#60a5fa;">
<span class="feat-title">PSI Parametric Laboratory (g / g<sub>e</sub>)</span>
</div>
<div class="feat-card-body">
Interactive maps and theoretical curves to compute extinction diameters, burning rates and flame stability as gravitational acceleration varies continuously.
</div>
</div>
""")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    f4, f5, f6 = st.columns(3)
    with f4:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#fb7185;">
<span class="feat-title">Artemis MPH Lunar Module (0.165 g<sub>e</sub>)</span>
</div>
<div class="feat-card-body">
Multiphysics simulator of fire spread in the pressurized lunar habitat (0.165 g<sub>e</sub>). Includes ventilation flows, polymer kinetics and toxic gases.
</div>
</div>
""")

    with f5:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#34d399;">
<span class="feat-title">NASA-STD Safety Standards</span>
</div>
<div class="feat-card-body">
Automatic compliance screening of simulations and tests against NASA-STD-3001 and NASA-STD-6001: O<sub>2</sub> thresholds, air velocity, toxicity and safety margins.
</div>
</div>
""")

    with f6:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#c084fc;">
<span class="feat-title">NASA Scientific Reports & Papers</span>
</div>
<div class="feat-card-body">
Hybrid semantic search (Chroma + BM25) over more than 320 pages of technical literature, an original page viewer and structured summaries for each experiment.
</div>
</div>
""")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # 5. Guida Rapida in 4 Step
    st.markdown("### Quick Start in 4 Steps")
    st.caption("How to run a complete aerospace fire safety analysis with EFESTO.")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 1</div>
<div class="step-title">Filter the Mission</div>
<div class="step-desc">
In the Dashboard, select the NASA PSI experiment code (e.g. <code>PSI-117</code>) or keep the overall view.
</div>
</div>
""")

    with s2:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 2</div>
<div class="step-title">Simulate and Compute</div>
<div class="step-desc">
Open the <b>PSI</b> simulator or the <b>Artemis MPH</b> module in the operational tabs to model combustion at lunar gravity.
</div>
</div>
""")

    with s3:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 3</div>
<div class="step-title">Ask the AI</div>
<div class="step-desc">
Ask the agent questions in natural language (e.g. <i>"What is the extinction diameter for heptane at 1 atm?"</i>) and get answers and charts.
</div>
</div>
""")

    with s4:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 4</div>
<div class="step-title">Verify and Export</div>
<div class="step-desc">
Check NASA-STD compliance in the Safety Standards tab and export tables and charts directly to CSV or Excel.
</div>
</div>
""")

    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)

    # 6. Call To Action Finale (stesso pulsante unico)
    b_col1, b_col2, b_col3 = st.columns([1, 2.5, 1])
    with b_col2:
        st.button("Enter the EFESTO Dashboard", type="primary", on_click=go_to_dashboard, key="bottom_enter_cta", width="stretch")

    # 7. Crediti del Team
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    html_clean("""
<div class="card" style="border-color: rgba(56, 189, 248, 0.2); background: linear-gradient(135deg, #070b14, #0f172a); margin-top: 1rem;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
<div>
<div style="font-weight:700; color:#f8fafc; font-size:1rem; display:flex; align-items:center; gap:8px;">
EFESTO — Flame in Freefall Platform
</div>
<div style="font-size:0.83rem; color:#94a3b8; margin-top:0.25rem;">
Designed and developed to provide engineering support for fire safety in pressurized lunar habitats (Artemis Program).<br>
Open scientific and experimental data courtesy of the <b>NASA Physical Sciences Informatics (PSI) System</b>.
</div>
</div>
<div style="text-align:right;">
<div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#38bdf8; font-weight:600;">
Version: 2.4.0 · Artemis & PSI-117
</div>
<div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem;">
Stack: Streamlit · DuckDB · Chroma · Claude Haiku · Plotly
</div>
</div>
</div>
</div>
""")


# ===========================================================================
# VISTA 2 — DASHBOARD OPERATIVA (AMBIENTI SEPARATI, NESSUNA TAB ANNIDATA)
# ===========================================================================
def render_dashboard():
    # BARRA DI CONTROLLO SUPERIORE CENTRATA (Sostituisce la sidebar)
    top_c1, top_c2, top_c3 = st.columns([1.6, 4.4, 2.2], vertical_alignment="center")
    with top_c1:
        if st.button("← Back to Home", key="top_back_home", width="stretch"):
            go_to_home()
            st.rerun()

    with top_c2:
        html_clean("""
        <div style="display:flex; align-items:center; gap:10px;">
          <img src="{logo}" alt="NAVI logo" style="width:52px; height:52px; border-radius:50%; box-shadow:0 0 0 1px rgba(203,213,225,0.25);">
          <div>
            <h3 style="margin:0; font-size:1.35rem; font-weight:800; background:linear-gradient(90deg, #f59e0b 0%, #ef4444 40%, #38bdf8 100%); -webkit-background-clip:text; color:transparent; line-height:1.2;">
              EFESTO — Operations Dashboard
            </h3>
            <div style="font-size:0.75rem; color:#94a3b8; font-family:'JetBrains Mono',monospace;">Flame in Freefall · NASA Microgravity Fire Safety</div>
          </div>
        </div>
        """.replace("{logo}", logo_data_uri()))

    with top_c3:
        selected_tag = st.selectbox(
            "NASA PSI experiment filter",
            [ALL] + store.experiment_tags,
            key="exp_tag",
            label_visibility="collapsed",
            help="Filter datasets, series and searches by NASA PSI experiment code."
        )

    tag = None if selected_tag == ALL else selected_tag

    def filt(df: pd.DataFrame, col: str = "experiment_tag") -> pd.DataFrame:
        return df if tag is None else df[df[col] == tag]

    series_f = filt(store.series)
    points_f = filt(store.points)
    ds_f = filt(store.datasets)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # =======================================================================
    # AMBIENTI SEPARATI DI PRIMO LIVELLO (UNA TAB DEDICATA PER CIASCUN MODULO)
    # =======================================================================
    (
        tab_agent,
        tab_data,
        tab_psi,
        tab_mph,
        tab_reports,
        tab_safety
    ) = st.tabs(
        ALL_DASHBOARD_TABS,
        default=TAB_AGENT,
        key="dashboard_active_tab",
        on_change="rerun"
    )

    # -----------------------------------------------------------------------
    # TAB 1: CO-PILOTA IA (Ambiente dedicato esclusivamente all'assistente)
    # -----------------------------------------------------------------------
    with tab_agent:
        st.markdown("#### EFESTO AI Engineering Co-pilot")
        st.caption("A dedicated space for conversational queries on NASA PSI physical data and scientific documentation. "
                   "It writes sandboxed SQL queries, generates Plotly charts and cites report pages.")

        c_chat_head1, c_chat_head2 = st.columns([5, 1.8], vertical_alignment="center")
        with c_chat_head1:
            st.markdown(f"**Active experiment for the agent:** `{selected_tag}`")
        with c_chat_head2:
            if st.button("New conversation", key="agent_tab_clear_chat", width="stretch"):
                st.session_state.messages, st.session_state.api_history = [], []
                st.rerun()

        chat_box = st.container()
        with chat_box:
            if not st.session_state.messages:
                st.info("**Co-pilot ready.** Try asking:\n"
                        "- *'Compare the experimental and simulated extinction diameter for heptane as pressure varies'*\n"
                        "- *'Explain the cool flame phenomenon observed in microgravity'* \n"
                        "- *'What are the NASA-STD safety limits for the oxygen fraction?'*")
            for i, m in enumerate(st.session_state.messages):
                with st.chat_message(m["role"]):
                    if m["role"] == "assistant":
                        render_assistant(m, f"m{i}")
                    else:
                        st.markdown(m["content"])

        if not api_key:
            st.warning("`ANTHROPIC_API_KEY` not found in `st.secrets`: the AI assistant is disabled. The simulators, experimental series and reports are still available.")

        query = st.chat_input("Ask the AI assistant a question (e.g. compare the experimental and simulated extinction diameter)", disabled=not api_key)

        if query and api_key:
            agent = FlameAgent(get_client(api_key, secret("ANTHROPIC_WORKSPACE_ID")), store, index, model,
                               effort=secret("ANTHROPIC_EFFORT"), safety=safety, safety_index=safety_index)
            st.session_state.messages.append({"role": "user", "content": query})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(query)
                with st.chat_message("assistant"):
                    with st.status("EFESTO is analyzing the NASA data and preparing the answer…", expanded=True) as status:
                        try:
                            result = agent.run(st.session_state.api_history, query, tag, on_event=status.write)
                            status.update(label="Analysis complete", state="complete", expanded=False)
                        except anthropic.AuthenticationError:
                            status.update(label="Invalid API key", state="error")
                            result = None
                        except anthropic.RateLimitError:
                            status.update(label="Rate limit reached: try again shortly", state="error")
                            result = None
                        except anthropic.APIConnectionError:
                            status.update(label="Connection error with the Anthropic API", state="error")
                            result = None
                        except anthropic.APIStatusError as e:
                            status.update(label=f"API error ({e.status_code})", state="error")
                            result = None

                    if result is not None:
                        msg = to_message(result)
                        render_assistant(msg, f"m{len(st.session_state.messages)}")
                        st.session_state.messages.append(msg)
                        st.session_state.api_history += [{"role": "user", "content": query},
                                                         {"role": "assistant", "content": FlameAgent.history_entry(result)}]
                        first = next((g for g in msg["explores"] if tag in (None, g["experiment_tag"])), None)
                        if first:
                            st.session_state.explore_target = first
                    else:
                        st.session_state.messages.pop()

    # -----------------------------------------------------------------------
    # TAB 2: DATI & SERIE SPERIMENTALI (Ambiente grafici ed esplorazione)
    # -----------------------------------------------------------------------
    with tab_data:
        st.markdown("#### NASA PSI Experimental Data & Series")
        st.caption("A dedicated space for visual analysis of historical series, multivariable comparison and data export.")

        if ds_f.empty:
            st.info("No datasets available for the selected filter.")
        else:
            series_count_by_ds = series_f.groupby("dataset").size().to_dict() if not series_f.empty else {}

            def format_dataset_label(r):
                n_ser = series_count_by_ds.get(r.dataset, 0)
                tag_label = f" [{n_ser} series]" if n_ser > 0 else " [metadata/images]"
                fig_str = f" · {r.figure}" if pd.notna(r.figure) and r.figure else ""
                pub_str = f" ({r.journal} {r.year})" if pd.notna(r.journal) and r.journal else ""
                return f"{r.dataset}{fig_str}{pub_str}{tag_label}"

            labels = {r.dataset: format_dataset_label(r) for r in ds_f.itertuples()}
            inv = {v: k for k, v in labels.items()}

            target = st.session_state.pop("explore_target", None)
            if target and target["dataset"] in labels:
                st.session_state.dash_ds = labels[target["dataset"]]
            if st.session_state.get("dash_ds") not in inv:
                st.session_state.pop("dash_ds", None)

            c1, c2 = st.columns([3, 2])
            chosen_label = c1.selectbox("Select a dataset to explore", list(inv), key="dash_ds")
            dataset = inv[chosen_label]
            s_ds = series_f[series_f.dataset == dataset] if not series_f.empty else pd.DataFrame()

            if s_ds.empty:
                st.info("**Metadata / photo archive / test log dataset (e.g. SAFFIRE-III sequences).**\n\n"
                        "This file contains metadata, calibration logs and space image sequences without plottable scalar x/y series. "
                        "You can still browse or download the full original table below as CSV or Excel.")
            else:
                yvars = s_ds.y_var.value_counts().index.tolist()
                ydesc = s_ds.drop_duplicates("y_var").set_index("y_var").y_description.to_dict()
                ylab = {f"{v} — {ydesc.get(v, '')}": v for v in yvars}
                if target and target["dataset"] == dataset and target["y_var"] in yvars:
                    st.session_state.dash_y = next(k for k, v in ylab.items() if v == target["y_var"])
                if st.session_state.get("dash_y") not in ylab:
                    st.session_state.pop("dash_y", None)
                y_var = ylab[c2.selectbox("Measured quantity (y axis)", list(ylab), key="dash_y")]
                s_sel = s_ds[s_ds.y_var == y_var]
                opts = s_sel.series_id.tolist()
                fmt = s_sel.set_index("series_id").apply(
                    lambda r: f"{r.name.split(':')[-1]} · {r.series_label} ({KIND_LABELS.get(r.kind, r.kind)}) · {r.n_points} pt", axis=1
                ).to_dict()

                c3, c4, c5 = st.columns([4, 1, 1])
                slab = {fmt[o]: o for o in opts}
                if target and target["dataset"] == dataset and target["y_var"] == y_var:
                    st.session_state.dash_sel = [fmt[i] for i in target["series_ids"] if i in fmt][:8]
                    st.session_state.dash_logx, st.session_state.dash_logy = target["log_x"], target["log_y"]
                    st.success("View set automatically from the AI assistant's suggestion.")
                elif st.session_state.get("dash_sel_ctx") != (dataset, y_var) or "dash_sel" not in st.session_state:
                    st.session_state.dash_sel = list(slab)[:8]
                st.session_state.dash_sel_ctx = (dataset, y_var)
                st.session_state.dash_sel = [x for x in st.session_state.dash_sel if x in slab]
                chosen = [slab[x] for x in c3.multiselect("Series to display", list(slab), max_selections=8, key="dash_sel")]
                log_x = c4.toggle("log x", key="dash_logx")
                log_y = c5.toggle("log y", key="dash_logy")

                if chosen:
                    pts = points_f[points_f.series_id.isin(chosen)]
                    if not pts.empty:
                        st.plotly_chart(series_chart(pts, log_x, log_y, title=f"{dataset}: {ydesc.get(y_var, y_var)}"),
                                        width="stretch", key="dash_series")
                        info = ds_f.set_index("dataset").loc[dataset]
                        st.caption(f"Source: `{info.source_file}` [{info.experiment_tag}]"
                                   + (f" — {info.journal} {info.year}, {info.figure} · [DOI]({info.reference_doi})" if info.reference_doi else ""))

                        unit = pts.y_unit.iloc[0] if pd.notna(pts.y_unit.iloc[0]) else ""
                        m = st.columns(4)
                        m[0].metric("Selected points", len(pts))
                        m[1].metric("y min", f"{pts.y.min():.4g} {unit}")
                        m[2].metric("y max", f"{pts.y.max():.4g} {unit}")
                        m[3].metric("y mean", f"{pts.y.mean():.4g} {unit}")

                        with st.expander("Selected points table", expanded=False):
                            cols = ["series_id", "series_label", "kind", "fuel", "pressure_atm", "d0_mm", "x_var", "x_unit", "x",
                                    "y_var", "y_unit", "y", "source_file", "x_column", "y_column"]
                            st.dataframe(pts[cols], width="stretch", hide_index=True, height=360)
                            download_buttons(pts[cols], f"{dataset}_{y_var}", "dash_pts", {"sources": [info.source_file], "DOI": info.reference_doi})
                    else:
                        st.warning("No points available for the selected series.")

            with st.expander("Full original table (NASA format, CSV / Excel download)"):
                raw = store.raw_table(dataset)
                st.dataframe(raw, width="stretch", hide_index=True, height=360)
                download_buttons(raw, f"original_{dataset}", "dash_raw", {"source file": labels[dataset]})

    # -----------------------------------------------------------------------
    # TAB 3: SIMULATORE PSI (Ambiente simulatore parametrico)
    # -----------------------------------------------------------------------
    with tab_psi:
        html_clean("<h4>NASA PSI Laboratory • Interactive Maps & Parametric Simulations (g / g<sub>e</sub>)</h4>")
        st.caption("Multi-experiment parametric simulator based on the Physical Sciences Informatics Repository (PSI-117, PSI-25, PSI-142, PSI-164, BASS, Saffire).")
        components.html(PSI_DASHBOARD_HTML, height=1150, scrolling=True)

    # -----------------------------------------------------------------------
    # TAB 4: MODULO LUNARE MPH (Ambiente simulatore modulo lunare)
    # -----------------------------------------------------------------------
    with tab_mph:
        html_clean("<h4>Artemis MPH • Fire Spread Simulator at Lunar Gravity (0.165 g<sub>e</sub>)</h4>")
        st.caption("Multiphysics aerospace simulator for the pressurized lunar habitat module (Multi-Purpose Habitat).")
        components.html(MPH_DASHBOARD_HTML, height=1150, scrolling=True)

    # -----------------------------------------------------------------------
    # TAB 5: REPORT SCIENTIFICI PSI (Ambiente documentazione & ricerca)
    # -----------------------------------------------------------------------
    with tab_reports:
        st.markdown("#### NASA PSI Scientific Reports")
        st.caption("Browse the original publications, analytical summaries and semantic search (BM25 + Chroma) with a page viewer.")

        docs_f = filt(store.documents)
        exp_f = filt(store.experiments) if not store.experiments.empty else store.experiments
        for e in exp_f.itertuples():
            if e.overview:
                st.markdown(f"<div class='card'><b>{e.experiment_tag}</b><br>{plain(e.overview)}</div>", unsafe_allow_html=True)
        st.subheader("Documents & Publications")
        for r in docs_f.itertuples():
            with st.expander(f"{r.doc_type} · {plain(r.title)} — {r.n_pages} pages"):
                st.caption(f"`{r.file_name}` · {r.experiment_tag}" + (f" · {plain(r.author)}" if r.author else "")
                           + f" · summary {r.summary_source}")
                st.markdown(plain(r.summary))
                findings = json.loads(r.key_findings_json or "[]")
                if findings:
                    st.markdown("**Key findings**")
                    for f in findings:
                        st.markdown(f"- {plain(f['finding'])} _(pp. {', '.join(map(str, f['pages']))})_")
                topics = json.loads(r.topics_json or "[]")
                if topics:
                    st.markdown("".join(f"<span class='pill'>{t}</span>" for t in topics), unsafe_allow_html=True)
                if r.experimental_conditions:
                    st.markdown(f"**Experimental conditions:** {plain(r.experimental_conditions)}")
                outline = json.loads(r.outline_json or "[]")
                if outline:
                    st.markdown("**Table of contents**")
                    st.dataframe(pd.DataFrame(outline).rename(columns={"section": "Section", "start_page": "Page"}),
                                 hide_index=True, width="stretch", height=min(300, 38 + 35 * len(outline)))

        st.subheader("Search and read the reports")
        c1, c2 = st.columns([3, 2])
        q = c1.text_input("Search the reports (semantic + keywords)", placeholder="cool flame extinction diameter", key="docs_q")
        doc_opts = docs_f.file_name.tolist()
        doc_sel = c2.selectbox("Specific document", ["All"] + doc_opts, key="docs_sel")
        if q:
            hits = index.search(q, experiment_tag=tag, file_name=None if doc_sel == "All" else doc_sel, top_k=8)
            for h in hits:
                st.markdown(f"**{h.file_name}** · p. {h.page_number}" + (f" · _{h.section}_" if h.section else "")
                            + f" <span class='pill'>{h.matched_by}</span>", unsafe_allow_html=True)
                st.caption(h.text[:500].replace("\n", " ") + "…")
        if doc_sel != "All":
            n_pages = int(docs_f.set_index("file_name").loc[doc_sel, "n_pages"])
            page = st.number_input("Page", 1, n_pages, 1, key="docs_page") if n_pages > 1 else 1
            row = index.read_pages(doc_sel, int(page), int(page))
            if not row.empty:
                r = row.iloc[0]
                st.caption(f"Section: {r.section or 'n/a'} · [Source: {r.experiment_tag} | {doc_sel}, p. {page}]")
                st.markdown(f"<div class='page-text'>{r.text.replace('<', '&lt;')}</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # TAB 6: NORME DI SICUREZZA NASA (Ambiente verifica standard astronautici)
    # -----------------------------------------------------------------------
    with tab_safety:
        st.markdown("#### NASA-STD Safety Standards (3001 & 6001)")
        st.caption("Automatic aerospace compliance checks: design limits, series verification and simulation file screening.")

        def show_checks(res: pd.DataFrame, key: str, title: str):
            if res.empty:
                st.info("No quantity matches a limit in the catalog.")
                return
            view = res.assign(result=res.esito.map(ESITO_ICON).fillna(res.esito))
            if "confronto" in view.columns:
                view = view.rename(columns={"confronto": "comparison"})
            cols = [c for c in ("result", "label", "source", "column", "parameter", "limit_id", "limit", "limit_type",
                                "value_min", "value_max", "n_violations", "worst_value", "margin", "comparison", "unit_note",
                                "citation") if c in view.columns]
            m = st.columns(3)
            m[0].metric("Checks", len(res))
            m[1].metric("Non-compliant", int(((res.esito == "non conforme")).sum()))
            m[2].metric("In risk zone", int(((res.esito == "in zona di rischio")).sum()))
            st.dataframe(view[cols], width="stretch", hide_index=True, height=min(420, 38 + 35 * len(view)))
            download_buttons(view[cols], title, key, {"limits source": "data/safety/safety_limits.parquet"})

        if safety is None:
            st.warning("Safety standards database not available. Build it with `python build_safety_db.py`."
                       + (f" \nDetails: {safety_error}" if safety_error else ""))
        else:
            st.subheader("Numerical Design Limits")
            lim = safety.limits
            lim_view = lim.assign(limit=[format_limit(r) for r in lim.itertuples()],
                                  automatic=lim.get("applies_by_default", True))
            st.dataframe(lim_view[["limit_id", "parameter", "limit", "limit_type", "scope", "automatic", "citation"]],
                         width="stretch", hide_index=True, height=min(420, 38 + 35 * len(lim_view)))
            with st.expander("Original text and notes for each limit"):
                for r in lim.itertuples():
                    st.markdown(f"**{r.limit_id}** — {r.description}")
                    st.caption(f"“{r.quote}” {r.citation}" + ("" if r.value_in_text else " · value not stated numerically in the text"))
            if not safety.requirements.empty:
                with st.expander(f"NASA-STD-3001 Qualitative Requirements ({len(safety.requirements)})"):
                    for r in safety.requirements.itertuples():
                        st.markdown(f"**[{r.requirement_id}] {r.title}** — {r.text}")
                        st.caption(r.citation)

            st.subheader("Verification of Stored Simulation Series")
            c1, c2, c3 = st.columns([3, 1, 1])
            kinds = c1.multiselect("Series types", list(KIND_LABELS), default=["simulation", "model"],
                                   format_func=lambda k: KIND_LABELS.get(k, k), key="saf_kinds")
            include_all = c2.toggle("Optional limits", key="saf_all", help="Include narrow-scope limits (inerted volumes, device capacities)")
            allow_proxy = c3.toggle("Proxy", key="saf_proxy", help="Indicative comparisons with related quantities (e.g. flame_length vs burn_length)")
            if kinds:
                res = safety.check_store(store, kinds=kinds, include_all=include_all, allow_proxy=allow_proxy)
                if tag is not None and not res.empty and "experiment_tag" in res:
                    res = res[res.experiment_tag == tag]
                show_checks(res, "saf_store", "standards_check_series")

            st.markdown("**Check a user simulation file** (CSV or Excel)")
            st.caption("Automatic column detection: `temperature_K`, `O2 [%]`, `x_o2`, `air_velocity_m_s`, `burn_length_mm`...")
            up = st.file_uploader("Upload simulation file", type=["csv", "xlsx"], key="saf_upload")
            if up is not None:
                try:
                    sim = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                    res = safety.check_any(sim, include_all=include_all, allow_proxy=allow_proxy)
                    show_checks(res, "saf_upload_res", f"standards_check_{up.name.rsplit('.', 1)[0]}")
                except Exception as e:
                    st.error(f"Unable to check the file: {e}")

            st.subheader("Search and read the NASA-STD standards")
            c1, c2 = st.columns([3, 2])
            sq = c1.text_input("Search the standards (semantic + keywords)", placeholder="oxygen concentration flammability limit", key="saf_q")
            sdocs = sorted(safety_index.pages.file_name.unique())
            sdoc = c2.selectbox("Standard document", ["All"] + sdocs, key="saf_doc")
            if sq:
                for h in safety_index.search(sq, file_name=None if sdoc == "All" else sdoc, top_k=8):
                    st.markdown(f"**{h.file_name}** · p. {h.page_number}" + (f" · _{h.section}_" if h.section else "")
                                + f" <span class='pill'>{h.matched_by}</span>", unsafe_allow_html=True)
                    st.caption(h.text[:500].replace("\n", " ") + "…")
            if sdoc != "All":
                n_pages = int(safety_index.pages[safety_index.pages.file_name == sdoc].page_number.max())
                page = st.number_input("Standard page", 1, n_pages, 1, key="saf_page") if n_pages > 1 else 1
                row = safety_index.read_pages(sdoc, int(page), int(page))
                if not row.empty:
                    r = row.iloc[0]
                    st.caption(f"Section: {r.section or 'n/a'} · [Source: {r.experiment_tag} | {sdoc}, p. {page}]")
                    st.markdown(f"<div class='page-text'>{r.text.replace('<', '&lt;')}</div>", unsafe_allow_html=True)


# ===========================================================================
# ROUTING DELL'APPLICAZIONE (HOME vs DASHBOARD)
# ===========================================================================
if st.session_state.current_page == "home":
    render_home_page()
else:
    render_dashboard()
