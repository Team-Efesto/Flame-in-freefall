import json

import anthropic
import pandas as pd
import plotly.express as px
import streamlit as st

from flame.agent import FlameAgent
from flame.charts import PALETTE, series_chart, style
from flame.config import DEFAULT_MODEL
from flame.data_store import DataStore
from flame.export import safe_filename, to_csv_bytes, to_xlsx_bytes
from flame.retrieval import ReportIndex

st.set_page_config(page_title="Flame in Freefall - NASA PSI", page_icon="🔥", layout="wide")

ALL = "Tutti gli esperimenti"
KIND_LABELS = {"experiment": "Esperimento", "model": "Modello", "simulation": "Simulazione", "unspecified": "Non specificato"}
KIND_COLORS = {"Esperimento": PALETTE[0], "Modello": PALETTE[1], "Simulazione": PALETTE[2], "Non specificato": "#6B7280"}
EXAMPLES = [
    "Come varia il diametro di estinzione della cool flame con la pressione? Confronta esperimento e modello.",
    "Estrai una tabella con tutte le misure sperimentali di estinzione per n-eptano, n-decano e n-dodecano.",
    "Riassumi i risultati principali della tesi di dottorato e cita le pagine.",
    "Cos'è una cool flame e perché in microgravità è più facile osservarla?",
]

# --------------------------------------------------------------------------- stile
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(255,107,44,0.12), transparent 60%),
        radial-gradient(900px 400px at -10% 110%, rgba(64,120,255,0.10), transparent 60%),
        #0B0F1A;
}
.block-container { padding-top: 2.5rem; max-width: 1280px; }
.hero {
    padding: 1.8rem 2.2rem; border-radius: 16px; border: 1px solid #232B42; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(255,107,44,0.14) 0%, rgba(20,26,43,0.9) 45%, rgba(20,26,43,0.9) 100%);
}
.hero-kicker { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #FF9A5C; margin-bottom: 0.5rem; }
.hero h1 {
    margin: 0 0 0.6rem 0; padding: 0; font-size: 2.3rem; line-height: 1.15;
    background: linear-gradient(90deg, #FFD3A8, #FF6B2C 55%, #FF3D3D);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 0; color: #A9B3C9; font-size: 1.02rem; max-width: 820px; }
section[data-testid="stSidebar"] { background: #0F1422; border-right: 1px solid #232B42; }
div[data-testid="stExpander"] details { border: 1px solid #232B42; border-radius: 12px; background: rgba(20,26,43,0.6); }
div[data-testid="stExpander"] summary:hover { color: #FF9A5C; }
div[data-testid="stChatMessage"] { border: 1px solid #232B42; border-radius: 14px; background: rgba(20,26,43,0.55); padding: 1rem 1.2rem; }
div[data-testid="stChatInput"] > div { border-radius: 14px; border-color: #2E3856; }
div[data-testid="stChatInput"] > div:focus-within { border-color: #FF6B2C; box-shadow: 0 0 0 3px rgba(255,107,44,0.18); }
div[data-testid="stMetric"] { border: 1px solid #232B42; border-radius: 12px; background: rgba(20,26,43,0.6); padding: 0.8rem 1rem; }
div[data-testid="stMetricValue"] { color: #FFD3A8; }
.card { border: 1px solid #232B42; border-radius: 12px; background: rgba(20,26,43,0.6); padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.pill { display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; border: 1px solid #2E3856; color: #A9B3C9; font-size: 0.78rem; margin: 0 0.3rem 0.3rem 0; }
.page-text { white-space: pre-wrap; font-size: 0.88rem; color: #C9D1E3; max-height: 520px; overflow-y: auto;
             border: 1px solid #232B42; border-radius: 10px; padding: 0.9rem 1rem; background: rgba(11,15,26,0.6); }
hr { border-color: #232B42 !important; }
code { color: #FFB27D !important; background: rgba(255,107,44,0.10) !important; border-radius: 6px; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">NASA PSI · Microgravity Combustion Science</div>
  <h1>🔥 Flame in Freefall</h1>
  <p>Piattaforma scientifica interattiva: interroga in linguaggio naturale l'intero dataset sperimentale e i report
  NASA PSI, ottieni grafici e tabelle scaricabili generati al volo, con citazione puntuale delle fonti.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- risorse
@st.cache_resource(show_spinner="Caricamento dataset…")
def load_store() -> DataStore:
    return DataStore()


@st.cache_resource(show_spinner="Caricamento indice dei report…")
def load_index(_store: DataStore) -> ReportIndex:
    return ReportIndex(_store.pages)


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
    c1.download_button("⬇️ CSV", to_csv_bytes(df), safe_filename(title, "csv"), "text/csv",
                       key=f"{key}_csv", on_click="ignore", width="stretch")
    c2.download_button("⬇️ Excel", xlsx_bytes(df, title, json.dumps(meta or {}, default=str)), safe_filename(title, "xlsx"),
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"{key}_xlsx", on_click="ignore", width="stretch")


store = load_store()
index = load_index(store)
ds_info = store.datasets.set_index("source_file")

# --------------------------------------------------------------------------- sidebar
st.sidebar.header("🔎 Filtro esperimento")
selected_tag = st.sidebar.selectbox("Codice NASA PSI", [ALL] + store.experiment_tags,
                                    help="Filtra dashboard, report e (come contesto) le ricerche dell'assistente.")
tag = None if selected_tag == ALL else selected_tag

api_key = secret("ANTHROPIC_API_KEY")
model = secret("ANTHROPIC_MODEL", DEFAULT_MODEL)
st.sidebar.divider()
st.sidebar.caption(f"🤖 Modello: `{model}`")
if index.collection is None:
    st.sidebar.caption(f"⚠️ Ricerca semantica non disponibile, uso solo parole chiave. ({index.vector_error})")
if st.sidebar.button("🗑️ Nuova conversazione", width="stretch"):
    st.session_state.messages, st.session_state.api_history = [], []
    st.rerun()
st.sidebar.caption("Dati: NASA Physical Science Informatics (PSI). I valori tabellari sono i dati delle figure degli "
                   "articoli citati (DOI nel catalogo).")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("api_history", [])


def filt(df: pd.DataFrame, col: str = "experiment_tag") -> pd.DataFrame:
    return df if tag is None else df[df[col] == tag]


# --------------------------------------------------------------------------- rendering risposte
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
    for i, ch in enumerate(msg.get("charts", [])):
        st.plotly_chart(ch["figure"], width="stretch", key=f"{key}_chart_{i}")
    for t in msg.get("tables", []):
        st.markdown(f"**📋 {t['title']}** · {len(t['df'])} righe" + (" · ⚠️ troncata" if t.get("truncated") else ""))
        st.dataframe(t["df"], width="stretch", height=min(420, 38 + 35 * len(t["df"])), hide_index=True)
        download_buttons(t["df"], t["title"], f"{key}_{t['id']}",
                         {"query SQL": t.get("sql") or "tabella originale", "fonti": t.get("sources", [])})
    render_sources(msg)
    if msg.get("trace"):
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
    }


# --------------------------------------------------------------------------- tabs
tab_chat, tab_dash, tab_docs, tab_cat = st.tabs(["💬 Assistente IA", "📊 Dashboard dati", "📚 Report scientifici", "🗂️ Catalogo & export"])

# ============================== ASSISTENTE ==============================
with tab_chat:
    chat_box = st.container()
    with chat_box:
        if not st.session_state.messages:
            st.markdown("##### Prova a chiedere")
        for i, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    render_assistant(m, f"m{i}")
                else:
                    st.markdown(m["content"])

    example = None
    if not st.session_state.messages:
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"ex{i}", width="stretch"):
                example = q

    if not api_key:
        st.warning("Chiave `ANTHROPIC_API_KEY` mancante in `st.secrets`: l'assistente è disattivato, "
                   "dashboard e report restano consultabili.")
    query = st.chat_input("Es: confronta il diametro di estinzione sperimentale e simulato al variare della pressione",
                          disabled=not api_key) or example

    if query and api_key:
        agent = FlameAgent(get_client(api_key, secret("ANTHROPIC_WORKSPACE_ID")), store, index, model,
                           effort=secret("ANTHROPIC_EFFORT"))
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
                else:
                    st.session_state.messages.pop()

# ============================== DASHBOARD ==============================
with tab_dash:
    series_f, points_f, ds_f = filt(store.series), filt(store.points), filt(store.datasets)
    docs_f = filt(store.documents)
    k = st.columns(5)
    k[0].metric("Esperimenti PSI", len(set(ds_f.experiment_tag) | set(docs_f.experiment_tag)))
    k[1].metric("Dataset / figure", int((ds_f.role == "figure_data").sum()))
    k[2].metric("Serie", len(series_f))
    k[3].metric("Punti dati", f"{len(points_f):,}".replace(",", "."))
    k[4].metric("Report (pagine)", f"{len(docs_f)} ({int(docs_f.n_pages.sum())})")

    if series_f.empty:
        st.info("Nessun dato tabellare per questo esperimento.")
    else:
        ov = series_f.assign(tipo=series_f.kind.map(KIND_LABELS)).groupby(["dataset", "tipo"], as_index=False).n_points.sum()
        fig = px.bar(ov, x="n_points", y="dataset", color="tipo", orientation="h", color_discrete_map=KIND_COLORS,
                     labels={"n_points": "Punti dati", "dataset": "", "tipo": ""})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.3, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig, "Punti dati per dataset e tipo di serie", height=380), width="stretch", key="ov_bar")

        st.subheader("🔬 Esplora le serie")
        fig_ds = ds_f[(ds_f.role == "figure_data") & ds_f.dataset.isin(series_f.dataset)]  # solo figure con serie
        labels = {r.dataset: f"{r.dataset} — {r.figure or ''} {('(' + r.journal + ' ' + str(r.year) + ')') if r.journal else ''}"
                  for r in fig_ds.itertuples()}
        c1, c2 = st.columns([3, 2])
        inv = {v: k for k, v in labels.items()}
        dataset = inv[c1.selectbox("Dataset (figura)", list(inv))]
        s_ds = series_f[series_f.dataset == dataset]
        yvars = s_ds.y_var.value_counts().index.tolist()
        ydesc = s_ds.drop_duplicates("y_var").set_index("y_var").y_description.to_dict()
        ylab = {f"{v} — {ydesc.get(v, '')}": v for v in yvars}
        y_var = ylab[c2.selectbox("Grandezza (asse y)", list(ylab))]
        s_sel = s_ds[s_ds.y_var == y_var]
        opts = s_sel.series_id.tolist()
        fmt = s_sel.set_index("series_id").apply(lambda r: f"{r.name.split(':')[-1]} · {r.series_label} ({KIND_LABELS[r.kind]}) · {r.n_points} pt", axis=1).to_dict()
        c3, c4, c5 = st.columns([4, 1, 1])
        slab = {fmt[o]: o for o in opts}
        chosen = [slab[x] for x in c3.multiselect("Serie", list(slab), default=list(slab)[:8], max_selections=8)]
        log_x = c4.toggle("log x")
        log_y = c5.toggle("log y")
        if chosen:
            pts = points_f[points_f.series_id.isin(chosen)]
            st.plotly_chart(series_chart(pts, log_x, log_y, title=f"{dataset}: {ydesc.get(y_var, y_var)}"),
                            width="stretch", key="dash_series")
            info = fig_ds.set_index("dataset").loc[dataset]
            st.caption(f"Fonte: `{info.source_file}` [{info.experiment_tag}]"
                       + (f" — {info.journal} {info.year}, {info.figure} · [DOI]({info.reference_doi})" if info.reference_doi else ""))
            m = st.columns(4)
            m[0].metric("Punti selezionati", len(pts))
            m[1].metric("y min", f"{pts.y.min():.4g} {pts.y_unit.iloc[0]}")
            m[2].metric("y max", f"{pts.y.max():.4g} {pts.y_unit.iloc[0]}")
            m[3].metric("y medio", f"{pts.y.mean():.4g} {pts.y_unit.iloc[0]}")
            with st.expander("📋 Tabella dei punti selezionati", expanded=False):
                cols = ["series_id", "series_label", "kind", "fuel", "pressure_atm", "d0_mm", "x_var", "x_unit", "x",
                        "y_var", "y_unit", "y", "source_file", "x_column", "y_column"]
                st.dataframe(pts[cols], width="stretch", hide_index=True, height=360)
                download_buttons(pts[cols], f"{dataset}_{y_var}", "dash_pts", {"fonti": [info.source_file], "DOI": info.reference_doi})
        with st.expander("📄 Tabella originale (formato del file NASA)"):
            raw = store.raw_table(dataset)
            st.dataframe(raw, width="stretch", hide_index=True, height=360)
            download_buttons(raw, f"originale_{dataset}", "dash_raw", {"file sorgente": labels[dataset]})

# ============================== REPORT ==============================
with tab_docs:
    exp_f = filt(store.experiments) if not store.experiments.empty else store.experiments
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
        page = st.number_input("Pagina", 1, n_pages, 1)
        row = index.read_pages(doc_sel, int(page), int(page))
        if not row.empty:
            r = row.iloc[0]
            st.caption(f"Sezione: {r.section or 'n/d'} · [Fonte: {r.experiment_tag} | {doc_sel}, p. {page}]")
            st.markdown(f"<div class='page-text'>{r.text.replace('<', '&lt;')}</div>", unsafe_allow_html=True)

# ============================== CATALOGO ==============================
with tab_cat:
    st.subheader("Dataset e figure")
    show = ds_f.drop(columns=["raw_rows", "raw_columns"])
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"reference_doi": st.column_config.LinkColumn("DOI"),
                                "n_points": st.column_config.NumberColumn("Punti"),
                                "n_series": st.column_config.NumberColumn("Serie")})
    download_buttons(show, "catalogo_dataset", "cat_ds")

    st.subheader("Catalogo delle serie")
    kinv = {v: k for k, v in KIND_LABELS.items()}
    kinds = [kinv[x] for x in st.multiselect("Tipo", list(kinv), default=list(kinv))]
    scat = series_f[series_f.kind.isin(kinds)]
    st.dataframe(scat, hide_index=True, width="stretch", height=380)
    download_buttons(scat, "catalogo_serie", "cat_series")

    st.subheader("Dataset completo (formato lungo)")
    st.caption(f"{len(points_f):,} punti — una riga per punto (x, y) con serie, condizioni, unità e file sorgente.".replace(",", "."))
    lazy_download_buttons(points_f, f"flame_in_freefall_punti_{tag or 'tutti'}", "cat_points",
                     {"fonti": sorted(points_f.source_file.unique())})
