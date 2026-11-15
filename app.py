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

st.set_page_config(page_title="EFESTO — NASA Microgravity Fire Safety Dashboard", page_icon="🔥", layout="wide")

ALL = "Tutti gli esperimenti"
KIND_LABELS = {"experiment": "Esperimento", "model": "Modello", "simulation": "Simulazione", "unspecified": "Non specificato"}

ESITO_ICON = {
    "conforme": "✅ conforme",
    "non conforme": "❌ non conforme",
    "fuori zona di rischio": "✅ fuori zona di rischio",
    "in zona di rischio": "⚠️ in zona di rischio"
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
    background-color: #000000;
    background-image: none;
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
    background: #000000;
}

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
    align-items: center;
    gap: 0.6rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.6rem;
}
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
::-webkit-scrollbar-track { background: #000000; }
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

/* Indicatore di caricamento con icone fucina e spazio */
[data-testid="stStatusWidgetRunningManIcon"], [data-testid="stStatusWidgetNewYearsIcon"] { display: none !important; }
[data-testid="stStatusWidgetRunningIcon"] { display: inline-flex; align-items: center; justify-content: center; }
[data-testid="stStatusWidgetRunningIcon"]::after {
    content: "⚒️"; font-size: 1.35rem; line-height: 1; display: inline-block;
    animation: efesto-icons 4.8s steps(1, end) infinite, efesto-float 1.2s ease-in-out infinite;
}
@keyframes efesto-icons {
    0%    { content: "⚒️"; }  16.6% { content: "🔨"; } 33.3%   { content: "🔥"; } 50% { content: "🚀"; }
    66.6% { content: "🪐"; }  83.3% { content: "🌙"; }
}
@keyframes efesto-float { 0%, 100% { transform: translateY(0) rotate(-6deg); } 50% { transform: translateY(-3px) rotate(6deg); } }
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
    c1.download_button("⬇️ CSV", csv_bytes(df), safe_filename(title, "csv"), "text/csv",
                       key=f"{key}_csv", on_click="ignore", width="stretch")
    c2.download_button("⬇️ Excel", xlsx_bytes(df, title, json.dumps(meta or {}, default=str)), safe_filename(title, "xlsx"),
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
TAB_AGENT = "💬 Co-pilota IA"
TAB_DATA = "📊 Dati & Serie Sperimentali"
TAB_PSI = "🪐 Simulatore PSI (g/gₑ)"
TAB_MPH = "🌙 Modulo Lunare MPH (0.165 gₑ)"
TAB_REPORTS = "📚 Report Scientifici PSI"
TAB_SAFETY = "🛡️ Norme di Sicurezza NASA"
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
    st.markdown("**📊 Grafici di questi dati disponibili nella scheda Dati & Serie**")
    for i, g in enumerate(guides):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.markdown(f"**{g['figure'] or g['dataset']}** — {g['y_description']} · {len(g['series_ids'])} serie")
        c2.button("Visualizza nella scheda Dati →", key=f"{key}_explore_{i}", on_click=apply_explore, args=(g,),
                  width="stretch", help=f"Dataset {g['dataset']} · grandezza {g['y_var']}")


def render_sources(msg: dict):
    hits, files = msg.get("doc_hits", []), msg.get("data_sources", [])
    if not hits and not files:
        return
    with st.expander(f"📚 Fonti consultate ({len(hits)} pagine di report, {len(files)} file di dati)"):
        if files:
            st.markdown("**Dati tabellari**")
            for f in files:
                info = ds_info.loc[f] if f in ds_info.index else None
                extra = ""
                if info is not None and pd.notna(info.get("reference_doi")):
                    extra = f" — {info['journal']} {info['year']}, {info['figure']} · [DOI]({info['reference_doi']})"
                st.markdown(f"- `{f}`{extra}")
        if hits:
            st.markdown("**Report**")
            for h in hits:
                st.markdown(f"- `{h['tag']}` — `{h['file']}` · **p. {h['page']}**" + (f" · _{h['section']}_" if h["section"] else ""))
                st.caption(h["text"][:300].replace("\n", " ") + "…")


def render_assistant(msg: dict, key: str):
    st.markdown(msg["content"])
    render_explore(msg.get("explores", []), key)
    for i, ch in enumerate(msg.get("charts", [])):
        st.plotly_chart(ch["figure"], width="stretch", key=f"{key}_chart_{i}")
    for t in msg.get("tables", []):
        st.markdown(f"**📋 {t['title']}** · {len(t['df'])} righe" + (" · ⚠️ troncata" if t.get("truncated") else ""))
        st.dataframe(t["df"], width="stretch", height=min(420, 38 + 35 * len(t["df"])), hide_index=True)
        download_buttons(t["df"], t["title"], f"{key}_{t['id']}",
                         {"query SQL": t.get("sql") or "tabella originale", "fonti": t.get("sources", [])})
    render_sources(msg)
    if msg.get("trace") and secret("DEBUG_MODE", False):
        with st.expander("🛠️ Dettagli dell'analisi"):
            for step in msg["trace"]:
                st.markdown(f"{'✅' if step['ok'] else '❌'} {step['label']}")
                if step["tool"] == "query_data":
                    st.code(step["input"].get("sql", ""), language="sql")
                if not step["ok"]:
                    st.caption(f"Errore (corretto automaticamente dal modello se possibile): {step.get('error')}")
            u = msg.get("usage") or {}
            st.caption(f"Token — input: {u.get('input_tokens', 0):,} · da cache: {u.get('cache_read_input_tokens', 0):,} · "
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
<div style="font-size:3rem; margin-bottom:0.5rem;">🔥</div>
<h3 style="color:#38bdf8; margin:0 0 0.5rem 0;">NASA FLEX-2 • Microgravity Droplet Combustion</h3>
<p style="color:#94a3b8; font-size:0.9rem; max-width:620px; margin:0 auto;">
Combustione di goccia isolata in microgravità (g<sub>eff</sub> ≈ 10<sup>-5</sup> g<sub>e</sub>). In assenza di galleggiamento termico la fiamma assume una simmetria perfettamente sferica dominata da trasporto diffusivo puro e regimi di cool flame.
</p>
</div>
"""
        html_clean(fallback_html)

    st.caption("🔬 **Esperimento NASA FLEX (Flame Extinction Experiment) • ISS Combustion Integrated Rack (CIR)** — "
               "Struttura sferica stazionaria di fiamma su goccia di idrocarburo isolata sospesa su fibre di quarzo. "
               "In microgravità pura (Ra ≈ 0), la convezione naturale è completamente soppressa: la fiamma brucia in condizioni diffusive pure "
               "e rivela la transizione a fiamme fredde (*cool flames*) a ~550 K prima dell'estinzione radiativa finale.")


# ===========================================================================
# VISTA 1 — HOME PAGE (SCHERMATA INIZIALE PULITA, 1 SOLO PUNTO D'INGRESSO)
# ===========================================================================
def render_home_page():
    # 1. Hero Section
    html_clean("""
<div class="efesto-hero">
<div class="efesto-badge">⚒️ EFESTO · NASA PSI · LUNAR HABITAT FIRE SAFETY ENGINEERING</div>
<div class="efesto-title-text">EFESTO</div>
<div class="efesto-subtitle-text">Flame in Freefall — NASA Microgravity Fire Safety Dashboard</div>
<div class="efesto-desc-text">
Piattaforma di ingegneria multifisica e intelligenza artificiale avanzata per la sicurezza antincendio
nell'esplorazione spaziale profonda e negli habitat pressurizzati del programma Artemis.<br>
EFESTO fonde i dati aperti del <b>NASA Physical Sciences Informatics (PSI)</b> con simulazioni computazionali,
modelli di combustione di gocce in microgravità (0 g) e in gravità lunare parziale (0.165 g<sub>e</sub>), verifica rigorosa
delle normative <b>NASA-STD-3001</b> e <b>NASA-STD-6001</b>, e un <b>Co-pilota IA</b> autonomo con citazioni verificate a livello di singola pagina.
</div>
</div>
""")

    # UNICO PUNTO D'INGRESSO NELLA HERO
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2.5, 1])
    with c_btn2:
        st.button("🔨 Entra nella Dashboard EFESTO", type="primary", on_click=go_to_dashboard, key="hero_enter_cta", width="stretch")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Immagine NASA FLEX + Fisica del fuoco
    c_img, c_info = st.columns([1.15, 0.85], gap="large")
    with c_img:
        st.markdown("#### 🔬 Combustione di Goccia in Microgravità (NASA FLEX)")
        render_nasa_flame_visual()

    with c_info:
        st.markdown("#### 🌌 La Fisica del Fuoco a g ≈ 0")
        html_clean("""
<div class="feat-card" style="margin-bottom: 0.8rem;">
<div class="feat-card-header" style="color: #38bdf8;">
<span>🫧</span> Assenza di Galleggiamento Termico
</div>
<div class="feat-card-body">
Sulla Terra la spinta d'Archimede allunga la fiamma in una colonna convettiva ascendente.
In caduta libera (g ≈ 10<sup>-5</sup> g<sub>e</sub>), con numero di Rayleigh Ra ≈ 0,
il pennacchio scompare: la fiamma assume una <b>simmetria sferica perfetta</b> controllata unicamente da diffusione molecolare.
</div>
</div>

<div class="feat-card" style="margin-bottom: 0.8rem;">
<div class="feat-card-header" style="color: #f59e0b;">
<span>❄️</span> Regime a Bassa Temperatura (Cool Flames)
</div>
<div class="feat-card-body">
Nelle gocce di idrocarburo (es. n-eptano nell'esperimento FLEX), dopo l'apparente estinzione della fiamma visibile ad alta temperatura,
la reazione chimica prosegue in un regime a <b>500–750 K</b> quasi impercettibile ad occhio nudo, continuando a consumare combustibile.
</div>
</div>

<div class="feat-card">
<div class="feat-card-header" style="color: #ef4444;">
<span>🌙</span> Transizione alla Gravità Lunare (0.165 g<sub>e</sub>)
</div>
<div class="feat-card-body">
Sulla Luna la gravità ridotta attiva una convezione lenta ma insidiosa. Nei moduli abitativi come il Multi-Purpose Habitat (MPH),
i limiti di estinzione e la velocità di propagazione su cavi e tessuti differiscono radicalmente dai test eseguiti a terra.
</div>
</div>
""")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # 3. Sezione Obiettivi
    st.markdown("### 🎯 Obiettivi Scientifici & Ingegneristici di EFESTO")
    st.caption("Progettato per ingegneri aerospaziali, ricercatori di microgravità e responsabili della sicurezza delle missioni spaziali con equipaggio.")

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#38bdf8;">
<span>01</span> Combustione Diffusiva
</div>
<div class="feat-card-body">
Caratterizzare la cinetica chimica e l'estinzione radiativa in assenza totale di convezione naturale (0 g) per prevenire inneschi incontrollati nei laboratori orbitali.
</div>
</div>
""")

    with o2:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#f59e0b;">
<span>02</span> Dati PSI & Modelli
</div>
<div class="feat-card-body">
Correlare oltre 15.000 punti sperimentali dal repository NASA PSI (PSI-117, BASS, Saffire) con la legge del diametro quadro (d<sup>2</sup>-law) e modelli analitici.
</div>
</div>
""")

    with o3:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#10b981;">
<span>03</span> Standard NASA-STD
</div>
<div class="feat-card-body">
Validare automaticamente simulazioni e scenari operativi contro le prescrizioni vincolanti di NASA-STD-3001 (Human System) e NASA-STD-6001 (Flammability).
</div>
</div>
""")

    with o4:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#a855f7;">
<span>04</span> Co-pilota con RAG
</div>
<div class="feat-card-body">
Garantire un'interrogazione trasparente basata su evidenze con query SQL in sandbox, generazione dinamica di grafici Plotly e citazioni a livello di pagina.
</div>
</div>
""")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # 4. Card delle Funzionalità (schede pulite senza bottoni ridondanti)
    st.markdown("### ⚡ Funzionalità della Piattaforma")
    st.caption("Una suite completa di strumenti analitici, multifisici e bibliografici accessibile all'interno della Dashboard.")

    f1, f2, f3 = st.columns(3)
    with f1:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#38bdf8;">
<span>💬</span> Co-pilota IA & RAG Scientifico
</div>
<div class="feat-card-body">
Assistente ingegneristico connesso a DuckDB e ChromaDB. Interroga dataset sperimentali con SQL sicuro, crea grafici Plotly e riporta la pagina esatta del report NASA.
</div>
</div>
""")

    with f2:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#f59e0b;">
<span>📊</span> Dati & Serie Sperimentali
</div>
<div class="feat-card-body">
Esplora 67 serie temporali storiche da missioni spaziali con selezione multi-serie, assi logaritmici, statistiche aggregate ed esportazione in CSV o Excel.
</div>
</div>
""")

    with f3:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#60a5fa;">
<span>🪐</span> Laboratorio Parametrico PSI (g / g<sub>e</sub>)
</div>
<div class="feat-card-body">
Mappe interattive e curve teoriche per calcolare diametri di estinzione, tassi di combustione e stabilità di fiamma al variare continuo dell'accelerazione di gravità.
</div>
</div>
""")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    f4, f5, f6 = st.columns(3)
    with f4:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#fb7185;">
<span>🌙</span> Modulo Lunare Artemis MPH (0.165 g<sub>e</sub>)
</div>
<div class="feat-card-body">
Simulatore multifisico della propagazione del fuoco nell'habitat lunare pressurizzato (0.165 g<sub>e</sub>). Include flussi di ventilazione, cinetica su polimeri e gas tossici.
</div>
</div>
""")

    with f5:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#34d399;">
<span>🛡️</span> Norme di Sicurezza NASA-STD
</div>
<div class="feat-card-body">
Screening automatico di conformità per simulazioni e test contro le norme NASA-STD-3001 e NASA-STD-6001: soglie O<sub>2</sub>, velocità aria, tossicità e margini di sicurezza.
</div>
</div>
""")

    with f6:
        html_clean("""
<div class="feat-card">
<div class="feat-card-header" style="color:#c084fc;">
<span>📄</span> Report Scientifici & Paper NASA
</div>
<div class="feat-card-body">
Ricerca semantica ibrida (Chroma + BM25) su oltre 320 pagine di letteratura tecnica, visualizzatore di pagine originali e sommari strutturati per ogni esperimento.
</div>
</div>
""")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.divider()

    # 5. Guida Rapida in 4 Step
    st.markdown("### 🚀 Guida Rapida all'Uso in 4 Step")
    st.caption("Come condurre un'analisi di sicurezza antincendio aerospaziale completa con EFESTO.")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 1</div>
<div class="step-title">Filtra la Missione</div>
<div class="step-desc">
Nella Dashboard seleziona il codice esperimento NASA PSI (es. <code>PSI-117</code>) o mantieni la vista complessiva.
</div>
</div>
""")

    with s2:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 2</div>
<div class="step-title">Simula e Calcola</div>
<div class="step-desc">
Apri il simulatore <b>PSI</b> o il modulo <b>Artemis MPH</b> nelle schede operative per modellare la combustione a gravità lunare.
</div>
</div>
""")

    with s3:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 3</div>
<div class="step-title">Interroga l'IA</div>
<div class="step-desc">
Fai domande all'agente in linguaggio naturale (es. <i>"Qual è il diametro di estinzione per l'eptano a 1 atm?"</i>) con risposte e grafici.
</div>
</div>
""")

    with s4:
        html_clean("""
<div class="step-item">
<div class="step-badge">STEP 4</div>
<div class="step-title">Verifica ed Esporta</div>
<div class="step-desc">
Controlla la conformità con NASA-STD nella scheda Report & Norme ed esporta tabelle e grafici direttamente in CSV o Excel.
</div>
</div>
""")

    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)

    # 6. Call To Action Finale (stesso pulsante unico)
    b_col1, b_col2, b_col3 = st.columns([1, 2.5, 1])
    with b_col2:
        st.button("🔨 Entra nella Dashboard EFESTO", type="primary", on_click=go_to_dashboard, key="bottom_enter_cta", width="stretch")

    # 7. Crediti del Team
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    html_clean("""
<div class="card" style="border-color: rgba(56, 189, 248, 0.2); background: linear-gradient(135deg, #070b14, #0f172a); margin-top: 1rem;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
<div>
<div style="font-weight:700; color:#f8fafc; font-size:1rem; display:flex; align-items:center; gap:8px;">
<span>⚒️</span> EFESTO — Flame in Freefall Platform
</div>
<div style="font-size:0.83rem; color:#94a3b8; margin-top:0.25rem;">
Progettato e sviluppato per il supporto ingegneristico alla sicurezza antincendio negli habitat lunari pressurizzati (Programma Artemis).<br>
Dati scientifici e sperimentali aperti per gentile concessione del <b>NASA Physical Sciences Informatics (PSI) System</b>.
</div>
</div>
<div style="text-align:right;">
<div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#38bdf8; font-weight:600;">
Versione: 2.4.0 · Artemis & PSI-117
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
        if st.button("← Torna alla Home", key="top_back_home", width="stretch"):
            go_to_home()
            st.rerun()

    with top_c2:
        html_clean("""
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:1.9rem;">⚒️</span>
          <div>
            <h3 style="margin:0; font-size:1.35rem; font-weight:800; background:linear-gradient(90deg, #f59e0b 0%, #ef4444 40%, #38bdf8 100%); -webkit-background-clip:text; color:transparent; line-height:1.2;">
              EFESTO — Dashboard Operativa
            </h3>
            <div style="font-size:0.75rem; color:#94a3b8; font-family:'JetBrains Mono',monospace;">Flame in Freefall · NASA Microgravity Fire Safety</div>
          </div>
        </div>
        """)

    with top_c3:
        selected_tag = st.selectbox(
            "Filtro Esperimento NASA PSI",
            [ALL] + store.experiment_tags,
            key="exp_tag",
            label_visibility="collapsed",
            help="Filtra dataset, serie e ricerche per codice esperimento NASA PSI."
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
        st.markdown("#### 💬 Co-pilota Ingegneristico IA EFESTO")
        st.caption("Ambiente dedicato per l'interrogazione conversazionale avanzata sui dati fisici NASA PSI e sulla documentazione scientifica. "
                   "Formula query SQL sandbox, genera grafici Plotly e cita pagine dei report.")

        c_chat_head1, c_chat_head2 = st.columns([5, 1.8], vertical_alignment="center")
        with c_chat_head1:
            st.markdown(f"**Esperimento attivo per l'agente:** `{selected_tag}`")
        with c_chat_head2:
            if st.button("🗑️ Nuova conversazione", key="agent_tab_clear_chat", width="stretch"):
                st.session_state.messages, st.session_state.api_history = [], []
                st.rerun()

        chat_box = st.container()
        with chat_box:
            if not st.session_state.messages:
                st.info("💡 **Co-pilota pronto.** Prova a chiedere:\n"
                        "- *'Confronta il diametro di estinzione sperimentale e simulato per l'eptano al variare della pressione'*\n"
                        "- *'Spiegami il fenomeno delle cool flames osservato in microgravità'* \n"
                        "- *'Quali sono i limiti di sicurezza NASA-STD per la frazione di ossigeno?'*")
            for i, m in enumerate(st.session_state.messages):
                with st.chat_message(m["role"]):
                    if m["role"] == "assistant":
                        render_assistant(m, f"m{i}")
                    else:
                        st.markdown(m["content"])

        if not api_key:
            st.warning("Chiave `ANTHROPIC_API_KEY` non rilevata in `st.secrets`: l'assistente IA è disattivato. I simulatori, le serie sperimentali e i report rimangono consultabili.")

        query = st.chat_input("Fai una domanda all'assistente IA (es: confronta il diametro di estinzione sperimentale e simulato)", disabled=not api_key)

        if query and api_key:
            agent = FlameAgent(get_client(api_key, secret("ANTHROPIC_WORKSPACE_ID")), store, index, model,
                               effort=secret("ANTHROPIC_EFFORT"), safety=safety, safety_index=safety_index)
            st.session_state.messages.append({"role": "user", "content": query})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(query)
                with st.chat_message("assistant"):
                    with st.status("⚒️ EFESTO sta analizzando i dati NASA e preparando la risposta…", expanded=True) as status:
                        try:
                            result = agent.run(st.session_state.api_history, query, tag, on_event=status.write)
                            status.update(label="Analisi completata", state="complete", expanded=False)
                        except anthropic.AuthenticationError:
                            status.update(label="Chiave API non valida", state="error")
                            result = None
                        except anthropic.RateLimitError:
                            status.update(label="Limite di richieste raggiunto: riprova tra poco", state="error")
                            result = None
                        except anthropic.APIConnectionError:
                            status.update(label="Errore di connessione all'API Anthropic", state="error")
                            result = None
                        except anthropic.APIStatusError as e:
                            status.update(label=f"Errore API ({e.status_code})", state="error")
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
        st.markdown("#### 📊 Dati & Serie Sperimentali NASA PSI")
        st.caption("Ambiente dedicato per l'analisi visiva delle serie storiche, il confronto multivariabile e l'esportazione dati.")

        if ds_f.empty:
            st.info("Nessun dataset disponibile per il filtro selezionato.")
        else:
            series_count_by_ds = series_f.groupby("dataset").size().to_dict() if not series_f.empty else {}

            def format_dataset_label(r):
                n_ser = series_count_by_ds.get(r.dataset, 0)
                icon = "📈" if n_ser > 0 else "📁"
                tag_label = f" [{n_ser} serie]" if n_ser > 0 else " [metadati/immagini]"
                fig_str = f" · {r.figure}" if pd.notna(r.figure) and r.figure else ""
                pub_str = f" ({r.journal} {r.year})" if pd.notna(r.journal) and r.journal else ""
                return f"{icon} {r.dataset}{fig_str}{pub_str}{tag_label}"

            labels = {r.dataset: format_dataset_label(r) for r in ds_f.itertuples()}
            inv = {v: k for k, v in labels.items()}

            target = st.session_state.pop("explore_target", None)
            if target and target["dataset"] in labels:
                st.session_state.dash_ds = labels[target["dataset"]]
            if st.session_state.get("dash_ds") not in inv:
                st.session_state.pop("dash_ds", None)

            c1, c2 = st.columns([3, 2])
            chosen_label = c1.selectbox("Seleziona Dataset da esplorare", list(inv), key="dash_ds")
            dataset = inv[chosen_label]
            s_ds = series_f[series_f.dataset == dataset] if not series_f.empty else pd.DataFrame()

            if s_ds.empty:
                st.info("ℹ️ **Dataset di tipo metadati / archivio fotografico / log di test (es. sequenze SAFFIRE-III).**\n\n"
                        "Questo file raccoglie metadati, log di calibrazione e sequenze di immagini spaziali senza serie scalari x/y plottabili. "
                        "È comunque possibile consultare o scaricare l'intera tabella originale sottostante in formato CSV o Excel.")
            else:
                yvars = s_ds.y_var.value_counts().index.tolist()
                ydesc = s_ds.drop_duplicates("y_var").set_index("y_var").y_description.to_dict()
                ylab = {f"{v} — {ydesc.get(v, '')}": v for v in yvars}
                if target and target["dataset"] == dataset and target["y_var"] in yvars:
                    st.session_state.dash_y = next(k for k, v in ylab.items() if v == target["y_var"])
                if st.session_state.get("dash_y") not in ylab:
                    st.session_state.pop("dash_y", None)
                y_var = ylab[c2.selectbox("Grandezza misurata (asse y)", list(ylab), key="dash_y")]
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
                    st.success("🔬 Vista impostata automaticamente dal suggerimento dell'assistente IA.")
                elif st.session_state.get("dash_sel_ctx") != (dataset, y_var) or "dash_sel" not in st.session_state:
                    st.session_state.dash_sel = list(slab)[:8]
                st.session_state.dash_sel_ctx = (dataset, y_var)
                st.session_state.dash_sel = [x for x in st.session_state.dash_sel if x in slab]
                chosen = [slab[x] for x in c3.multiselect("Serie da visualizzare", list(slab), max_selections=8, key="dash_sel")]
                log_x = c4.toggle("log x", key="dash_logx")
                log_y = c5.toggle("log y", key="dash_logy")

                if chosen:
                    pts = points_f[points_f.series_id.isin(chosen)]
                    if not pts.empty:
                        st.plotly_chart(series_chart(pts, log_x, log_y, title=f"{dataset}: {ydesc.get(y_var, y_var)}"),
                                        width="stretch", key="dash_series")
                        info = ds_f.set_index("dataset").loc[dataset]
                        st.caption(f"Fonte: `{info.source_file}` [{info.experiment_tag}]"
                                   + (f" — {info.journal} {info.year}, {info.figure} · [DOI]({info.reference_doi})" if info.reference_doi else ""))

                        unit = pts.y_unit.iloc[0] if pd.notna(pts.y_unit.iloc[0]) else ""
                        m = st.columns(4)
                        m[0].metric("Punti selezionati", len(pts))
                        m[1].metric("y min", f"{pts.y.min():.4g} {unit}")
                        m[2].metric("y max", f"{pts.y.max():.4g} {unit}")
                        m[3].metric("y medio", f"{pts.y.mean():.4g} {unit}")

                        with st.expander("📋 Tabella dei punti selezionati", expanded=False):
                            cols = ["series_id", "series_label", "kind", "fuel", "pressure_atm", "d0_mm", "x_var", "x_unit", "x",
                                    "y_var", "y_unit", "y", "source_file", "x_column", "y_column"]
                            st.dataframe(pts[cols], width="stretch", hide_index=True, height=360)
                            download_buttons(pts[cols], f"{dataset}_{y_var}", "dash_pts", {"fonti": [info.source_file], "DOI": info.reference_doi})
                    else:
                        st.warning("Nessun punto disponibile per le serie selezionate.")

            with st.expander("📄 Tabella originale completa (formato NASA con download CSV / Excel)"):
                raw = store.raw_table(dataset)
                st.dataframe(raw, width="stretch", hide_index=True, height=360)
                download_buttons(raw, f"originale_{dataset}", "dash_raw", {"file sorgente": labels[dataset]})

    # -----------------------------------------------------------------------
    # TAB 3: SIMULATORE PSI (Ambiente simulatore parametrico)
    # -----------------------------------------------------------------------
    with tab_psi:
        html_clean("<h4>🪐 Laboratorio NASA PSI • Mappe Interattive & Simulazioni Parametriche (g / g<sub>e</sub>)</h4>")
        st.caption("Simulatore parametrico multi-esperimento basato sul Physical Sciences Informatics Repository (PSI-117, PSI-25, PSI-142, PSI-164, BASS, Saffire).")
        components.html(PSI_DASHBOARD_HTML, height=1150, scrolling=True)

    # -----------------------------------------------------------------------
    # TAB 4: MODULO LUNARE MPH (Ambiente simulatore modulo lunare)
    # -----------------------------------------------------------------------
    with tab_mph:
        html_clean("<h4>🌙 Artemis MPH • Simulatore Propagazione Incendio in Gravità Lunare (0.165 g<sub>e</sub>)</h4>")
        st.caption("Simulatore aerospaziale multifisico per il modulo abitativo lunare pressurizzato (Multi-Purpose Habitat).")
        components.html(MPH_DASHBOARD_HTML, height=1150, scrolling=True)

    # -----------------------------------------------------------------------
    # TAB 5: REPORT SCIENTIFICI PSI (Ambiente documentazione & ricerca)
    # -----------------------------------------------------------------------
    with tab_reports:
        st.markdown("#### 📚 Report Scientifici NASA PSI")
        st.caption("Consultazione delle pubblicazioni originali, sommari analitici e ricerca semantica (BM25 + Chroma) con visualizzatore di pagine.")

        docs_f = filt(store.documents)
        exp_f = filt(store.experiments) if not store.experiments.empty else store.experiments
        for e in exp_f.itertuples():
            if e.overview:
                st.markdown(f"<div class='card'><b>🧪 {e.experiment_tag}</b><br>{e.overview}</div>", unsafe_allow_html=True)
        st.subheader("📑 Documenti & Pubblicazioni")
        for r in docs_f.itertuples():
            with st.expander(f"{r.doc_type} · {r.title} — {r.n_pages} pagine"):
                st.caption(f"`{r.file_name}` · {r.experiment_tag}" + (f" · {r.author}" if r.author else "")
                           + f" · sommario {r.summary_source}")
                st.markdown(r.summary)
                findings = json.loads(r.key_findings_json or "[]")
                if findings:
                    st.markdown("**Risultati chiave**")
                    for f in findings:
                        st.markdown(f"- {f['finding']} _(pp. {', '.join(map(str, f['pages']))})_")
                topics = json.loads(r.topics_json or "[]")
                if topics:
                    st.markdown("".join(f"<span class='pill'>{t}</span>" for t in topics), unsafe_allow_html=True)
                if r.experimental_conditions:
                    st.markdown(f"**Condizioni sperimentali:** {r.experimental_conditions}")
                outline = json.loads(r.outline_json or "[]")
                if outline:
                    st.markdown("**Indice**")
                    st.dataframe(pd.DataFrame(outline).rename(columns={"section": "Sezione", "start_page": "Pagina"}),
                                 hide_index=True, width="stretch", height=min(300, 38 + 35 * len(outline)))

        st.subheader("🔎 Cerca e leggi nei report")
        c1, c2 = st.columns([3, 2])
        q = c1.text_input("Ricerca nei report (semantica + parole chiave in inglese)", placeholder="cool flame extinction diameter", key="docs_q")
        doc_opts = docs_f.file_name.tolist()
        doc_sel = c2.selectbox("Documento specifico", ["Tutti"] + doc_opts, key="docs_sel")
        if q:
            hits = index.search(q, experiment_tag=tag, file_name=None if doc_sel == "Tutti" else doc_sel, top_k=8)
            for h in hits:
                st.markdown(f"**{h.file_name}** · p. {h.page_number}" + (f" · _{h.section}_" if h.section else "")
                            + f" <span class='pill'>{h.matched_by}</span>", unsafe_allow_html=True)
                st.caption(h.text[:500].replace("\n", " ") + "…")
        if doc_sel != "Tutti":
            n_pages = int(docs_f.set_index("file_name").loc[doc_sel, "n_pages"])
            page = st.number_input("Pagina", 1, n_pages, 1, key="docs_page") if n_pages > 1 else 1
            row = index.read_pages(doc_sel, int(page), int(page))
            if not row.empty:
                r = row.iloc[0]
                st.caption(f"Sezione: {r.section or 'n/d'} · [Fonte: {r.experiment_tag} | {doc_sel}, p. {page}]")
                st.markdown(f"<div class='page-text'>{r.text.replace('<', '&lt;')}</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # TAB 6: NORME DI SICUREZZA NASA (Ambiente verifica standard astronautici)
    # -----------------------------------------------------------------------
    with tab_safety:
        st.markdown("#### 🛡️ Norme di Sicurezza NASA-STD (3001 & 6001)")
        st.caption("Ambiente per il controllo automatico di conformità aerospaziale: limiti di progetto, verifica serie e screening file di simulazione.")

        def show_checks(res: pd.DataFrame, key: str, title: str):
            if res.empty:
                st.info("Nessuna grandezza corrisponde a un limite del catalogo.")
                return
            view = res.assign(esito=res.esito.map(ESITO_ICON).fillna(res.esito))
            cols = [c for c in ("esito", "label", "source", "column", "parameter", "limit_id", "limit", "limit_type",
                                "value_min", "value_max", "n_violations", "worst_value", "margin", "confronto", "unit_note",
                                "citation") if c in view.columns]
            m = st.columns(3)
            m[0].metric("Verifiche", len(res))
            m[1].metric("Non conformi", int(((res.esito == "non conforme")).sum()))
            m[2].metric("In zona di rischio", int(((res.esito == "in zona di rischio")).sum()))
            st.dataframe(view[cols], width="stretch", hide_index=True, height=min(420, 38 + 35 * len(view)))
            download_buttons(res, title, key, {"fonte limiti": "data/safety/safety_limits.parquet"})

        if safety is None:
            st.warning("Database delle norme di sicurezza non disponibile. Generalo con `python build_safety_db.py`."
                       + (f" \nDettaglio: {safety_error}" if safety_error else ""))
        else:
            st.subheader("📏 Limiti Numerici di Progetto")
            lim = safety.limits
            lim_view = lim.assign(limite=[format_limit(r) for r in lim.itertuples()],
                                  automatico=lim.get("applies_by_default", True))
            st.dataframe(lim_view[["limit_id", "parameter", "limite", "limit_type", "scope", "automatico", "citation"]],
                         width="stretch", hide_index=True, height=min(420, 38 + 35 * len(lim_view)))
            with st.expander("Testo originale e note di ogni limite"):
                for r in lim.itertuples():
                    st.markdown(f"**{r.limit_id}** — {r.description}")
                    st.caption(f"“{r.quote}” {r.citation}" + ("" if r.value_in_text else " · valore non riportato numericamente nel testo"))
            if not safety.requirements.empty:
                with st.expander(f"📜 Requisiti Qualitativi NASA-STD-3001 ({len(safety.requirements)})"):
                    for r in safety.requirements.itertuples():
                        st.markdown(f"**[{r.requirement_id}] {r.title}** — {r.text}")
                        st.caption(r.citation)

            st.subheader("🧪 Verifica delle Serie di Simulazione dello Store")
            c1, c2, c3 = st.columns([3, 1, 1])
            kinds = c1.multiselect("Tipi di serie", list(KIND_LABELS), default=["simulation", "model"],
                                   format_func=lambda k: KIND_LABELS.get(k, k), key="saf_kinds")
            include_all = c2.toggle("Limiti opzionali", key="saf_all", help="Includi i limiti ad ambito ristretto (volumi inertizzati, capacità di dispositivi)")
            allow_proxy = c3.toggle("Proxy", key="saf_proxy", help="Confronti indicativi con grandezze affini (es. flame_length vs burn_length)")
            if kinds:
                res = safety.check_store(store, kinds=kinds, include_all=include_all, allow_proxy=allow_proxy)
                if tag is not None and not res.empty and "experiment_tag" in res:
                    res = res[res.experiment_tag == tag]
                show_checks(res, "saf_store", "verifica_norme_serie")

            st.markdown("**Verifica un file di simulazione utente** (CSV o Excel)")
            st.caption("Riconoscimento automatico colonne: `temperature_K`, `O2 [%]`, `x_o2`, `air_velocity_m_s`, `burn_length_mm`...")
            up = st.file_uploader("Carica file di simulazione", type=["csv", "xlsx"], key="saf_upload")
            if up is not None:
                try:
                    sim = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                    res = safety.check_any(sim, include_all=include_all, allow_proxy=allow_proxy)
                    show_checks(res, "saf_upload_res", f"verifica_norme_{up.name.rsplit('.', 1)[0]}")
                except Exception as e:
                    st.error(f"Impossibile verificare il file: {e}")

            st.subheader("🔎 Cerca e leggi nelle norme NASA-STD")
            c1, c2 = st.columns([3, 2])
            sq = c1.text_input("Ricerca nelle norme (semantica + parole chiave in inglese)", placeholder="oxygen concentration flammability limit", key="saf_q")
            sdocs = sorted(safety_index.pages.file_name.unique())
            sdoc = c2.selectbox("Documento normativo", ["Tutti"] + sdocs, key="saf_doc")
            if sq:
                for h in safety_index.search(sq, file_name=None if sdoc == "Tutti" else sdoc, top_k=8):
                    st.markdown(f"**{h.file_name}** · p. {h.page_number}" + (f" · _{h.section}_" if h.section else "")
                                + f" <span class='pill'>{h.matched_by}</span>", unsafe_allow_html=True)
                    st.caption(h.text[:500].replace("\n", " ") + "…")
            if sdoc != "Tutti":
                n_pages = int(safety_index.pages[safety_index.pages.file_name == sdoc].page_number.max())
                page = st.number_input("Pagina norma", 1, n_pages, 1, key="saf_page") if n_pages > 1 else 1
                row = safety_index.read_pages(sdoc, int(page), int(page))
                if not row.empty:
                    r = row.iloc[0]
                    st.caption(f"Sezione: {r.section or 'n/d'} · [Fonte: {r.experiment_tag} | {sdoc}, p. {page}]")
                    st.markdown(f"<div class='page-text'>{r.text.replace('<', '&lt;')}</div>", unsafe_allow_html=True)


# ===========================================================================
# ROUTING DELL'APPLICAZIONE (HOME vs DASHBOARD)
# ===========================================================================
if st.session_state.current_page == "home":
    render_home_page()
else:
    render_dashboard()
