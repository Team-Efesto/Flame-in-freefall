import json
from pathlib import Path

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

st.set_page_config(page_title="Flame in Freefall - NASA PSI", page_icon="🔥", layout="wide")

ALL = "Tutti gli esperimenti"
KIND_LABELS = {"experiment": "Esperimento", "model": "Modello", "simulation": "Simulazione", "unspecified": "Non specificato"}

TAB_CHAT = "💬 Assistente IA"
TAB_DASH = "📊 Dashboard dati"
TAB_PSI = "🪐 Simulatore PSI (g/g_e)"
TAB_DOCS = "📚 Report scientifici"

TAB_SAFETY = "🛡️ Norme di sicurezza"
ESITO_ICON = {"conforme": "✅ conforme", "non conforme": "❌ non conforme",
              "fuori zona di rischio": "✅ fuori zona di rischio", "in zona di rischio": "⚠️ in zona di rischio"}

HTML_DASHBOARD_FILE = Path(__file__).parent / "dashboard_psi.html"
with open(HTML_DASHBOARD_FILE, "r", encoding="utf-8") as f:
    PSI_DASHBOARD_HTML = f.read()

# --------------------------------------------------------------------------- stile NASA PSI
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

<style>
.stApp {
    background-color: #0b1120;
    background-image:
        linear-gradient(to right, rgba(30, 41, 59, 0.35) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(30, 41, 59, 0.35) 1px, transparent 1px);
    background-size: 32px 32px;
    color: #e2e8f0;
    font-family: 'Inter', system-ui, sans-serif;
}
::selection {
    background: rgba(56, 189, 248, 0.3);
    color: #bae6fd;
}
.block-container { padding-top: 2.2rem; max-width: 1320px; }
.hero {
    padding: 1.8rem 2.2rem; border-radius: 16px; border: 1px solid #1e293b; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(17, 24, 39, 0.95) 45%, rgba(15, 23, 42, 0.95) 100%);
    box-shadow: 0 4px 24px -4px rgba(0, 0, 0, 0.6), inset 0 1px 0 0 rgba(255, 255, 255, 0.04);
}
.hero-kicker {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 0.6rem; display: inline-block;
    padding: 0.2rem 0.65rem; border-radius: 6px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35);
}
.hero h1 {
    margin: 0 0 0.6rem 0; padding: 0; font-size: 2.3rem; font-weight: 800; line-height: 1.15;
    background: linear-gradient(90deg, #ffffff, #38bdf8 55%, #818cf8);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-family: 'Inter', sans-serif;
}

.hero p { margin: 0; color: #94a3b8; font-size: 0.98rem; max-width: 860px; line-height: 1.5; }
section[data-testid="stSidebar"] { background: #0b1120; border-right: 1px solid #1e293b; }
div[data-testid="stExpander"] details { border: 1px solid #1e293b; border-radius: 12px; background: #111827; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
div[data-testid="stExpander"] summary { font-family: 'Inter', sans-serif; color: #e2e8f0; font-weight: 600; }
div[data-testid="stExpander"] summary:hover { color: #38bdf8; }
div[data-testid="stChatMessage"] { border: 1px solid #1e293b; border-radius: 14px; background: #111827; padding: 1rem 1.2rem; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
div[data-testid="stChatInput"] > div { border-radius: 14px; border-color: #1e293b; background: #111827; color: #e2e8f0; }
div[data-testid="stChatInput"] > div:focus-within { border-color: #38bdf8; box-shadow: 0 0 0 3px rgba(56,189,248,0.25); }
div[data-testid="stMetric"] { border: 1px solid #1e293b; border-radius: 12px; background: #111827; padding: 0.8rem 1rem; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
div[data-testid="stMetricValue"] { color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
div[data-testid="stMetricLabel"] { color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 0.78rem; }
.card { border: 1px solid #1e293b; border-radius: 12px; background: #111827; padding: 1rem 1.2rem; margin-bottom: 0.8rem; box-shadow: 0 4px 20px -4px rgba(0,0,0,0.4); }
.pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.35); background: rgba(56, 189, 248, 0.12); color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; margin: 0 0.3rem 0.3rem 0; }
.page-text { white-space: pre-wrap; font-size: 0.86rem; color: #cbd5e1; max-height: 520px; overflow-y: auto;
             border: 1px solid #1e293b; border-radius: 10px; padding: 0.9rem 1rem; background: #070b14; font-family: 'JetBrains Mono', monospace; }
hr { border-color: #1e293b !important; }
code { color: #38bdf8 !important; background: rgba(56,189,248,0.12) !important; border: 1px solid rgba(56,189,248,0.25) !important; border-radius: 6px; font-family: 'JetBrains Mono', monospace !important; }
button[data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-size: 0.88rem; font-weight: 600; color: #94a3b8 !important; }
button[data-baseweb="tab"]:hover { color: #e2e8f0 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #0b1120; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }


#MainMenu, footer { visibility: hidden; }
/* indicatore di caricamento: icone spaziali al posto degli omini di Streamlit */
[data-testid="stStatusWidgetRunningManIcon"], [data-testid="stStatusWidgetNewYearsIcon"] { display: none !important; }
[data-testid="stStatusWidgetRunningIcon"] { display: inline-flex; align-items: center; justify-content: center; }
[data-testid="stStatusWidgetRunningIcon"]::after {
    content: "🚀"; font-size: 1.35rem; line-height: 1; display: inline-block;
    animation: space-icons 4.8s steps(1, end) infinite, space-float 1.2s ease-in-out infinite;
}
@keyframes space-icons {
    0%    { content: "🚀"; }  12.5% { content: "🛰️"; } 25%   { content: "🧑‍🚀"; } 37.5% { content: "🪐"; }
    50%   { content: "☄️"; }  62.5% { content: "🌍"; }  75%   { content: "🌙"; }  87.5% { content: "🔭"; }
}
@keyframes space-float { 0%, 100% { transform: translateY(0) rotate(-6deg); } 50% { transform: translateY(-3px) rotate(6deg); } }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <div class="hero-kicker">NASA PSI · Lunar Habitat Fire Safety Engineering</div>
  <h1>🔥 Flame in Freefall</h1>
  <p>Co-pilota ingegneristico per la sicurezza antincendio negli habitat lunari: correla dataset sperimentali e report
  NASA PSI per analizzare limiti di estinzione, pressioni e diluenti, con grafici e fonti verificate.</p>
</div>
""", unsafe_allow_html=True)
# segnaposto subito sotto il banner: qui compaiono i pulsanti delle sezioni (definiti piu' avanti, dopo i callback)
nav_slot = st.container()


# --------------------------------------------------------------------------- risorse
@st.cache_resource(show_spinner="Caricamento dataset…")
def load_store() -> DataStore:
    return DataStore()


@st.cache_resource(show_spinner="Caricamento indice dei report…")
def load_index(_store: DataStore) -> ReportIndex:
    return ReportIndex(_store.pages)


@st.cache_resource(show_spinner="Caricamento norme di sicurezza…")
def load_safety() -> tuple[SafetyLimits | None, ReportIndex | None, str | None]:
    """Limiti + indice delle norme antincendio; l'app funziona anche se il database non e' stato generato."""
    try:
        limits = SafetyLimits()
        pages = pd.read_parquet(SAFETY_DIR / "safety_pages.parquet")
        return limits, ReportIndex(pages, SAFETY_VECTOR_DB_DIR, collection=SAFETY_COLLECTION), None
    except Exception as e:  # es. build_safety_db.py non ancora eseguito
        return None, None, str(e)


def secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:  # nessun secrets.toml
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


def lazy_download_buttons(df: pd.DataFrame, title: str, key: str, meta: dict | None = None):
    """Per tabelle grandi: CSV/Excel generati solo su richiesta, non ad ogni rerun."""
    flag = f"prep_{key}"
    if st.button(f"Prepara download ({len(df):,} righe)".replace(",", "."), key=f"{key}_prep"):
        st.session_state[flag] = True
    if st.session_state.get(flag):
        download_buttons(df, title, key, meta)


def download_buttons(df: pd.DataFrame, title: str, key: str, meta: dict | None = None):
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("⬇️ CSV", csv_bytes(df), safe_filename(title, "csv"), "text/csv",
                       key=f"{key}_csv", on_click="ignore", width="stretch")
    c2.download_button("⬇️ Excel", xlsx_bytes(df, title, json.dumps(meta or {}, default=str)), safe_filename(title, "xlsx"),
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"{key}_xlsx", on_click="ignore", width="stretch")


store = load_store()
index = load_index(store)
safety, safety_index, safety_error = load_safety()
ds_info = store.datasets.set_index("source_file")

# --------------------------------------------------------------------------- sidebar
st.sidebar.header("🔎 Filtro esperimento")
selected_tag = st.sidebar.selectbox("Codice NASA PSI", [ALL] + store.experiment_tags, key="exp_tag",
                                    help="Filtra dashboard, report e (come contesto) le ricerche dell'assistente.")
tag = None if selected_tag == ALL else selected_tag

api_key = secret("ANTHROPIC_API_KEY")
model = secret("ANTHROPIC_MODEL", DEFAULT_MODEL)
st.sidebar.divider()
if st.sidebar.button("🪐 Apri Simulatore PSI (g/g_e)", width="stretch"):
    st.session_state.panel_open = True
    st.session_state.panel_wide = True
    st.session_state.main_tab = TAB_PSI
    st.rerun()
if st.sidebar.button("🗑️ Nuova conversazione", width="stretch"):
    st.session_state.messages, st.session_state.api_history = [], []
    st.rerun()
st.sidebar.info(
    "**Focus Missione:** Supporto decisionale per ingegneri di sicurezza antincendio (habitat lunari e moduli pressurizzati) "
    "basato su dati sperimentali e pubblicazioni **NASA PSI**."
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("api_history", [])


def filt(df: pd.DataFrame, col: str = "experiment_tag") -> pd.DataFrame:
    return df if tag is None else df[df[col] == tag]


# --------------------------------------------------------------------------- rendering risposte
def apply_explore(sugg: dict):
    """Callback: imposta 'Esplora le serie' con i dati della risposta e apre la Dashboard dati."""
    if st.session_state.get("exp_tag", ALL) not in (ALL, sugg["experiment_tag"]):
        st.session_state.exp_tag = ALL
    st.session_state.explore_target = sugg
    st.session_state.main_tab = TAB_DASH
    st.session_state.panel_open = True


def render_explore(guides: list[dict], key: str):
    if not guides:
        return
    with st.container(border=True):
        st.markdown("**📊 Grafici di questi dati nella Dashboard dati**")
        for i, g in enumerate(guides):
            c1, c2 = st.columns([3, 2], vertical_alignment="center")
            c1.markdown(f"**{g['figure'] or g['dataset']}** — {g['y_description']} · {len(g['series_ids'])} serie")
            c2.button("Mostra nel pannello →", key=f"{key}_explore_{i}", on_click=apply_explore, args=(g,),
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


# --------------------------------------------------------------------------- layout: chat al centro + pannello laterale a scomparsa
st.session_state.setdefault("panel_open", False)
st.session_state.setdefault("panel_wide", False)


def open_panel(tab: str):
    st.session_state.main_tab = tab
    st.session_state.panel_open = True


def close_panel():
    st.session_state.panel_open = False


def toggle_wide():
    st.session_state.panel_wide = not st.session_state.get("panel_wide", False)


# Il pannello e' sempre renderizzato (cosi' grafici e selezioni restano pronti) ma, se chiuso, viene spostato fuori schermo.

panel_w_css = "min(1680px, 96vw)" if st.session_state.get("panel_wide") else "min(880px, 62vw)"


st.markdown(f"""
<style>
.st-key-chat_area {{ max-width: 1000px; margin: 0 auto; }}
div[data-testid="stBottomBlockContainer"] {{ max-width: 1000px; margin: 0 auto; padding-bottom: 1.4rem; }}
div[data-testid="stBottom"] > div {{ background: #0b1120; }}
.st-key-side_panel {{
    position: fixed; top: 0; right: 0; height: 100vh; width: {panel_w_css}; z-index: 999990;
    overflow-y: auto; overflow-x: hidden; padding: 4.2rem 1.6rem 2rem 1.6rem; gap: 0.6rem;
    background: #0f172a; border-left: 1px solid #1e293b; box-shadow: -18px 0 40px rgba(0,0,0,0.6);
    transition: transform 0.25s ease, visibility 0.25s, width 0.25s ease;
    {"" if st.session_state.panel_open else "transform: translateX(105%); visibility: hidden;"}
}}
.st-key-panel_open_btn {{ position: fixed; top: 4.2rem; right: 1.2rem; z-index: 999989; width: auto !important; }}
.st-key-panel_open_btn button {{
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.35); border-color: #38bdf8;
    background: #111827; color: #38bdf8; font-weight: 600;
}}
.st-key-panel_open_btn button:hover {{ background: rgba(56, 189, 248, 0.15); border-color: #7dd3fc; }}
@media (max-width: 900px) {{ .st-key-side_panel {{ width: 100vw !important; padding: 4rem 16px 2rem 16px; }} }}
</style>
""", unsafe_allow_html=True)

if not st.session_state.panel_open:
    with st.container(key="panel_open_btn"):
        st.button("📊 Dashboard, simulatore e report", on_click=toggle_panel, type="primary")

# ============================== ASSISTENTE ==============================
# chat_input al livello principale: Streamlit lo fissa in fondo allo schermo, indipendente dallo scorrimento.
query = st.chat_input("Es: confronta il diametro di estinzione sperimentale e simulato al variare della pressione",
                      disabled=not api_key)

@st.cache_resource(show_spinner=False)
def dataset_kpis(_store: DataStore) -> dict:
    """Numeri di sintesi calcolati dal dataset caricato (non valori fissi)."""
    p = _store.points
    # pressione ambiente di prova: condizioni delle serie e assi di pressione (non le pressioni delle linee di bordo)
    amb = pd.concat([_store.series.pressure_atm, p.loc[p.x_var == "pressure", "x"]]).dropna()
    return {"experiments": len(_store.experiment_tags), "exp_points": int((p.kind == "experiment").sum()),
            "all_points": len(p), "reports": len(_store.documents), "pages": int(_store.documents.n_pages.sum()),
            "p_min": float(amb.min()) if len(amb) else None, "p_max": float(amb.max()) if len(amb) else None}


def it_num(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def it_dec(x: float) -> str:
    return f"{x:g}".replace(".", ",")


kpi = dataset_kpis(store)
with st.container(key="kpi_row"):
    k = st.columns(4)
    k[0].metric("Esperimenti NASA PSI", kpi["experiments"], help="Codici PSI distinti con dati o report caricati.")
    k[1].metric("Punti sperimentali", it_num(kpi["exp_points"]),
                help=f"Misure di tipo esperimento, su {it_num(kpi['all_points'])} punti totali (esperimenti, modelli e simulazioni).")
    k[2].metric("Report tecnici indicizzati", kpi["reports"], help=f"{it_num(kpi['pages'])} pagine consultabili dall'assistente.")
    if kpi["p_min"] is not None:
        k[3].metric("Range di pressione", f"{it_dec(kpi['p_min'])}–{it_dec(kpi['p_max'])} atm",
                    help=f"Pressione ambiente delle prove (≈ {it_num(kpi['p_min'] * 101.325)}–{it_num(kpi['p_max'] * 101.325)} kPa).")

with st.container(key="chat_area"):
    st.subheader(TAB_CHAT)
    chat_box = st.container()
    with chat_box:
        for i, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    render_assistant(m, f"m{i}")
                else:
                    st.markdown(m["content"])

    if not api_key:
        st.warning("Chiave `ANTHROPIC_API_KEY` mancante in `st.secrets`: l'assistente è disattivato, "
                   "dashboard e report restano consultabili.")

    if query and api_key:
        agent = FlameAgent(get_client(api_key, secret("ANTHROPIC_WORKSPACE_ID")), store, index, model,
                           effort=secret("ANTHROPIC_EFFORT"), safety=safety, safety_index=safety_index)
        st.session_state.messages.append({"role": "user", "content": query})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                with st.status("Analisi in corso…", expanded=True) as status:
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
                    # prepara nel pannello laterale il primo grafico dei dati consultati (senza aprirlo sopra la chat)
                    first = next((g for g in msg["explores"] if tag in (None, g["experiment_tag"])), None)
                    if first:
                        st.session_state.explore_target = first
                        st.session_state.main_tab = TAB_DASH
                else:
                    st.session_state.messages.pop()

# ============================== PANNELLO LATERALE ==============================
side_panel = st.container(key="side_panel")
with side_panel:

    h1, h2, h3 = st.columns([3, 1, 1], vertical_alignment="center")
    h1.markdown("#### 🔬 Dati, simulazioni e report")
    h2.button("⛶ Espandi" if not st.session_state.get("panel_wide") else "🗗 Riduci", key="panel_wide_btn", on_click=toggle_wide, width="stretch")
    h3.button("✕ Chiudi", key="panel_close_btn", on_click=toggle_panel, width="stretch")
    tab_dash, tab_psi, tab_docs, tab_safety = st.tabs([TAB_DASH, TAB_PSI, TAB_DOCS, TAB_SAFETY], key="main_tab", on_change="rerun")

with tab_dash:
    series_f, points_f, ds_f = filt(store.series), filt(store.points), filt(store.datasets)

    if series_f.empty:
        st.info("Nessun dato tabellare per questo esperimento.")
    else:
        st.subheader("🔬 Esplora le serie")
        fig_ds = ds_f[ds_f.role == "figure_data"]
        if fig_ds.empty:
            st.info("Nessuna figura tabellare disponibile per questo filtro.")
        else:
            labels = {r.dataset: f"{r.dataset} — {r.figure or ''} {('(' + r.journal + ' ' + str(r.year) + ')') if r.journal else ''}"
                      for r in fig_ds.itertuples()}
            c1, c2 = st.columns([3, 2])
            inv = {v: k for k, v in labels.items()}
            ss = st.session_state
            target = ss.pop("explore_target", None)  # suggerimento dell'assistente da applicare
            if target and target["dataset"] in labels:
                ss.dash_ds = labels[target["dataset"]]
            if ss.get("dash_ds") not in inv:
                ss.pop("dash_ds", None)
            dataset = inv[c1.selectbox("Dataset (figura)", list(inv), key="dash_ds")]
            s_ds = series_f[series_f.dataset == dataset]

            if s_ds.empty:
                st.info("Nessuna serie misurata per questo dataset.")
            else:
                yvars = s_ds.y_var.value_counts().index.tolist()
                ydesc = s_ds.drop_duplicates("y_var").set_index("y_var").y_description.to_dict()
                ylab = {f"{v} — {ydesc.get(v, '')}": v for v in yvars}
                if target and target["dataset"] == dataset and target["y_var"] in yvars:
                    ss.dash_y = next(k for k, v in ylab.items() if v == target["y_var"])
                if ss.get("dash_y") not in ylab:
                    ss.pop("dash_y", None)
                y_var = ylab[c2.selectbox("Grandezza (asse y)", list(ylab), key="dash_y")]
                s_sel = s_ds[s_ds.y_var == y_var]
                opts = s_sel.series_id.tolist()
                fmt = s_sel.set_index("series_id").apply(
                    lambda r: f"{r.name.split(':')[-1]} · {r.series_label} ({KIND_LABELS.get(r.kind, r.kind)}) · {r.n_points} pt", axis=1
                ).to_dict()

                c3, c4, c5 = st.columns([4, 1, 1])
                slab = {fmt[o]: o for o in opts}
                if target and target["dataset"] == dataset and target["y_var"] == y_var:
                    ss.dash_sel = [fmt[i] for i in target["series_ids"] if i in fmt][:8]
                    ss.dash_logx, ss.dash_logy = target["log_x"], target["log_y"]
                    st.success("🔬 Vista impostata dal suggerimento dell'assistente.")
                elif ss.get("dash_sel_ctx") != (dataset, y_var) or "dash_sel" not in ss:
                    ss.dash_sel = list(slab)[:8]  # nuova figura/grandezza: selezione predefinita
                ss.dash_sel_ctx = (dataset, y_var)
                ss.dash_sel = [x for x in ss.dash_sel if x in slab]
                chosen = [slab[x] for x in c3.multiselect("Serie", list(slab), max_selections=8, key="dash_sel")]
                log_x = c4.toggle("log x", key="dash_logx")
                log_y = c5.toggle("log y", key="dash_logy")
                
                if chosen:
                    pts = points_f[points_f.series_id.isin(chosen)]
                    if not pts.empty:
                        st.plotly_chart(series_chart(pts, log_x, log_y, title=f"{dataset}: {ydesc.get(y_var, y_var)}"),
                                        width="stretch", key="dash_series")
                        info = fig_ds.set_index("dataset").loc[dataset]
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
                        
            with st.expander("📄 Tabella originale (formato del file NASA)"):
                raw = store.raw_table(dataset)
                st.dataframe(raw, width="stretch", hide_index=True, height=360)
                download_buttons(raw, f"originale_{dataset}", "dash_raw", {"file sorgente": labels[dataset]})

# ============================== SIMULATORE PSI (g/g_e) ==============================
with tab_psi:
    st.markdown("### 🪐 Laboratorio NASA PSI • Mappe Interattive & Simulazioni Parametriche ($g / g_e$)")
    st.caption("Simulatore parametrico multi-esperimento basato sul Physical Sciences Informatics Repository (PSI-117, PSI-25, PSI-142, PSI-164, BASS, Saffire).")
    components.html(PSI_DASHBOARD_HTML, height=1150, scrolling=True)

# ============================== REPORT ==============================
with tab_docs:
    docs_f = filt(store.documents)
    exp_f =filt(store.experiments) if not store.experiments.empty else store.experiments
    for e in exp_f.itertuples():
        if e.overview:
            st.markdown(f"<div class='card'><b>🧪 {e.experiment_tag}</b><br>{e.overview}</div>", unsafe_allow_html=True)
    st.subheader("📑 Documenti")
    for r in docs_f.itertuples():
        with st.expander(f"{r.doc_type} · {r.title}  —  {r.n_pages} pagine"):
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
    q = c1.text_input("Ricerca (semantica + parole chiave; i report sono in inglese)", placeholder="cool flame extinction diameter")
    doc_opts = docs_f.file_name.tolist()
    doc_sel = c2.selectbox("Documento", ["Tutti"] + doc_opts)
    if q:
        hits = index.search(q, experiment_tag=tag, file_name=None if doc_sel == "Tutti" else doc_sel, top_k=8)
        for h in hits:
            st.markdown(f"**{h.file_name}** · p. {h.page_number}" + (f" · _{h.section}_" if h.section else "")
                        + f"  <span class='pill'>{h.matched_by}</span>", unsafe_allow_html=True)
            st.caption(h.text[:500].replace("\n", " ") + "…")
    if doc_sel != "Tutti":
        n_pages = int(docs_f.set_index("file_name").loc[doc_sel, "n_pages"])
        page = st.number_input("Pagina", 1, n_pages, 1) if n_pages > 1 else 1
        row = index.read_pages(doc_sel, int(page), int(page))
        
        if not row.empty:
            r = row.iloc[0]
            st.caption(f"Sezione: {r.section or 'n/d'} · [Fonte: {r.experiment_tag} | {doc_sel}, p. {page}]")
            st.markdown(f"<div class='page-text'>{r.text.replace('<', '&lt;')}</div>", unsafe_allow_html=True)

# ============================== SIMULATORE & TEST ==============================
with tab_sim:
    st.subheader("🧪 Simulatore & test")
    sim_conf = st.selectbox("Scelta configurazione", SIM_CONFIGS, index=None, key="sim_config",
                            format_func=lambda c: f"{SIM_CONFIGS.index(c) + 1}. {c}",
                            placeholder="Seleziona una configurazione")
    sim_w = st.number_input("Larghezza W [mm]", min_value=0.0, value=None, step=1.0, format="%.2f",
                            placeholder="Inserisci la larghezza del campione", key="sim_w")
    sim_p = st.number_input("Pressione ambiente P [kPa]", min_value=0.0, value=None, step=1.0, format="%.2f",
                            placeholder="Inserisci la pressione ambiente", key="sim_p")
    sim_u = st.number_input("Velocità ventola imposta U_forced [cm/s]", min_value=0.0, value=None, step=0.5, format="%.2f",
                            placeholder="Inserisci la velocità del flusso forzato", key="sim_u")
    sim_o2 = st.number_input("Concentrazione ossigeno O₂ [% vol]", min_value=0.0, max_value=100.0, value=None, step=0.5,
                             format="%.2f", placeholder="Inserisci la concentrazione di ossigeno", key="sim_o2")
    sim_t0 = st.number_input("Temperatura iniziale T0 [K]", min_value=0.0, value=None, step=1.0, format="%.2f",
                             placeholder="Inserisci la temperatura iniziale", key="sim_t0")

# ============================== NORME DI SICUREZZA ==============================
def show_checks(res: pd.DataFrame, key: str, title: str):
    """Tabella dei risultati di verifica con icone di esito e download."""
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


with tab_safety:
    if safety is None:
        st.warning("Database delle norme di sicurezza non disponibile. Generalo con `python build_safety_db.py`."
                   + (f"  \nDettaglio: {safety_error}" if safety_error else ""))
    else:
        st.caption("Norme e documenti NASA di sicurezza antincendio spaziale (database separato dai report PSI; "
                   "il filtro esperimento della barra laterale non si applica qui).")
        st.subheader("📏 Limiti numerici")
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
            with st.expander(f"📜 Requisiti NASA-STD-3001 ({len(safety.requirements)})"):
                for r in safety.requirements.itertuples():
                    st.markdown(f"**[{r.requirement_id}] {r.title}** — {r.text}")
                    st.caption(r.citation)
        st.caption("Per CO, HCN, HCl i documenti rimandano alle SMAC (JSC-20584), non incluse nel corpus.")

        st.subheader("🧪 Verifica delle simulazioni")
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

        st.markdown("**Verifica un file di simulazione** (CSV/XLSX)")
        st.caption("Le colonne vengono riconosciute dal nome con l'unità, es. `temperature_K`, `O2 [%]`, `x_o2`, "
                   "`air_velocity_m_s`, `burn_length_mm`; sono accettate anche tabelle nel formato lungo di `points`.")
        up = st.file_uploader("File dei risultati", type=["csv", "xlsx"], key="saf_upload")
        if up is not None:
            try:
                sim = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                res = safety.check_any(sim, include_all=include_all, allow_proxy=allow_proxy)
                show_checks(res, "saf_upload_res", f"verifica_norme_{up.name.rsplit('.', 1)[0]}")
            except Exception as e:
                st.error(f"Impossibile verificare il file: {e}")

        st.subheader("🔎 Cerca e leggi nelle norme")
        c1, c2 = st.columns([3, 2])
        sq = c1.text_input("Ricerca (semantica + parole chiave; documenti in inglese)", placeholder="oxygen concentration flammability limit", key="saf_q")
        sdocs = sorted(safety_index.pages.file_name.unique())
        sdoc = c2.selectbox("Documento", ["Tutti"] + sdocs, key="saf_doc")
        if sq:
            for h in safety_index.search(sq, file_name=None if sdoc == "Tutti" else sdoc, top_k=8):
                st.markdown(f"**{h.file_name}** · p. {h.page_number}" + (f" · _{h.section}_" if h.section else "")
                            + f"  <span class='pill'>{h.matched_by}</span>", unsafe_allow_html=True)
                st.caption(h.text[:500].replace("\n", " ") + "…")
        if sdoc != "Tutti":
            n_pages = int(safety_index.pages[safety_index.pages.file_name == sdoc].page_number.max())
            page = st.number_input("Pagina", 1, n_pages, 1, key="saf_page") if n_pages > 1 else 1
            row = safety_index.read_pages(sdoc, int(page), int(page))
            if not row.empty:
                r = row.iloc[0]
                st.caption(f"Sezione: {r.section or 'n/d'} · [Fonte: {r.experiment_tag} | {sdoc}, p. {page}]")
                st.markdown(f"<div class='page-text'>{r.text.replace('<', '&lt;')}</div>", unsafe_allow_html=True)
