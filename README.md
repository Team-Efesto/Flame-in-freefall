# 🔥 Flame in Freefall — NASA Microgravity Fire Safety Platform

**NASA Space Apps Challenge 2026**  
Piattaforma scientifica interattiva per esplorare, correlare e interrogare i dataset sperimentali e le pubblicazioni del database **NASA Physical Science Informatics (PSI)** sulla combustione e la sicurezza antincendio in microgravità.

---

## 🚀 Panoramica

I dati sperimentali (diametri di goccia e di fiamma, radianza, temperature, diametri di estinzione…) e i risultati scientifici (tesi, report, presentazioni) sono frammentati in file eterogenei. **Flame in Freefall** li unisce in un'unica piattaforma:

* **Assistente IA con tool use (Claude):** interroga l'*intero* dataset numerico con SQL (non pochi frammenti), legge i report con un retrieval ibrido e genera **al volo grafici interattivi e tabelle scaricabili** pertinenti alla domanda.
* **Citazioni obbligatorie:** ogni affermazione riporta codice `PSI-XXX`, nome del file e pagina (report) oppure file e serie (dati).
* **Dashboard analitica:** KPI, grafici riassuntivi, esplorazione delle serie, tabelle filtrate con export **CSV** ed **Excel**.
* **Report navigabili:** sommari strutturati per documento ed esperimento, indice delle sezioni, ricerca e lettura pagina per pagina.

---

## ✨ Funzionalità

| Sezione | Cosa offre |
|---|---|
| 💬 **Assistente IA** | Domande in linguaggio naturale. Domande quantitative → query SQL sul dataset completo + tabella scaricabile; trend/confronti → grafico Plotly; domande teoriche → risposta dai report senza grafici. Pannelli "Fonti consultate" e "Dettagli dell'analisi" (query SQL eseguite, token). |
| 📊 **Dashboard dati** | KPI, punti per dataset e tipo (esperimento / modello / simulazione), esploratore di serie con scala log, tabella dei punti e tabella originale NASA, download CSV/XLSX. |
| 📚 **Report scientifici** | Sintesi per esperimento, sommario e risultati chiave (con pagine) per ogni PDF, indice delle sezioni, ricerca ibrida, lettore di pagine. |
| 🗂️ **Catalogo & export** | Catalogo dataset (con DOI) e serie, export dell'intero dataset in formato lungo. |

Il filtro **Codice NASA PSI** nella barra laterale si applica a tutte le sezioni.

---

## 🧱 Architettura

```text
             offline (build)                                   online (Streamlit)
dati_nasa_raw/ ──► build_json.py ──► data/neural_network_dataset.json
                                            │
                                            ▼
              build_processed.py ──► data/processed/*.parquet ──► DataStore (DuckDB in memoria, SQL read-only)
                    │                (points, series, datasets,         │
                    │                 doc_pages, documents, …)          ├─► FlameAgent (Claude + tool use)
dati_nasa_raw/ ──► build_vector_db.py ──► vector_db/ (ChromaDB) ──► ReportIndex (semantico + BM25 + pagine)
                    └── patch metadati / sommari LLM (cache: summaries.json)
```

* **Dati tabellari** — le tabelle NASA sono "figure data" in formato largo (più serie x/y affiancate). `flame/tidy.py` le converte in un formato lungo (una riga per punto) con serie, tipo (experiment/model/simulation), combustibile, pressione, d0, grandezze canoniche e unità, mantenendo la tracciabilità alla colonna originale.
* **Query engine** — Claude usa lo strumento `query_data` (SQL DuckDB) sull'intero dataset. Sicurezza: una sola `SELECT`, parole chiave pericolose bloccate, `enable_external_access=false` + `lock_configuration=true` (nessun accesso a file o rete), limite righe. Al modello arrivano solo anteprime e statistiche; i risultati completi restano all'app per grafici e download.
* **RAG sui report** — retrieval ibrido (embeddings `all-MiniLM-L6-v2` + BM25, fusione RRF) a livello di **pagina** con sezione/capitolo; lettura di intervalli di pagine; sommari strutturati per documento e per esperimento generati offline (nessun costo a runtime).
* **Strumenti dell'agente** — `query_data`, `create_chart`, `search_reports`, `read_report_pages`, `get_document_overview`, `get_raw_table`.

---

## 🛠️ Stack

Streamlit · Anthropic Python SDK (Claude Haiku 5.5 con tool use e prompt caching) · DuckDB · Pandas/PyArrow (Parquet) · Plotly · ChromaDB (embeddings ONNX `all-MiniLM-L6-v2`, senza torch) · OpenPyXL · PyPDF

---

## 📂 Struttura

```text
Flame-in-freefall/
├── .streamlit/
│   ├── config.toml                  # Tema (versionato)
│   └── secrets.toml                 # Chiavi API locali (escluso da Git)
├── data/
│   ├── neural_network_dataset.json  # Dataset unificato dai CSV/XLSX NASA
│   └── processed/                   # Artefatti interrogabili (Parquet) + sommari
├── flame/                           # Pacchetto applicativo
│   ├── agent.py                     # Agente Claude con tool use
│   ├── charts.py                    # Grafici Plotly (palette accessibile, tema scuro)
│   ├── config.py                    # Percorsi e costanti
│   ├── data_store.py                # Parquet + SQL read-only (DuckDB)
│   ├── export.py                    # Export CSV / Excel con foglio metadati
│   ├── retrieval.py                 # Retrieval ibrido sui report
│   └── tidy.py                      # Formato largo → formato lungo
├── vector_db/                       # ChromaDB pre-indicizzato
├── app.py                           # Applicazione Streamlit
├── build_json.py                    # (offline) CSV/XLSX → JSON
├── build_vector_db.py               # (offline) PDF → ChromaDB
├── build_processed.py               # (offline) JSON/vector DB → data/processed
├── ARCHITECTURE_PROGRESS.md         # Stato dei lavori e scelte architetturali
└── requirements.txt
```

---

## ⚙️ Installazione e avvio in locale

Richiede **Python 3.11+**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Crea `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
# Opzionali
ANTHROPIC_WORKSPACE_ID = "..."
ANTHROPIC_MODEL = "claude-haiku-5-5"
ANTHROPIC_EFFORT = "medium"     # low | medium | high
```

Avvio:
```bash
streamlit run app.py
```
Senza chiave API la dashboard e i report restano consultabili; l'assistente si disattiva.

### Rigenerazione dei dati (opzionale)
Gli artefatti in `data/processed/` e `vector_db/` sono già versionati. Per aggiornarli con nuovi file NASA (nomi contenenti `PSI-XXX`) in `./dati_nasa_raw/tabelle` e `./dati_nasa_raw/report_pdf`:
```bash
python build_json.py
python build_vector_db.py --rebuild     # ricrea l'indice dei PDF
python build_processed.py --summaries   # tabelle, pagine/sezioni, sommari (usa la chiave API una sola volta)
```
`build_processed.py` funziona anche senza la cartella grezza: ricostruisce tutto dal JSON e dal vector DB esistenti.

### Deploy su Streamlit Community Cloud
Imposta `ANTHROPIC_API_KEY` (e opzionalmente `ANTHROPIC_WORKSPACE_ID`) nei *Secrets* dell'app. Al primo avvio viene scaricato una sola volta il modello di embedding ONNX (~80 MB).

---

## 👥 Il Team

Progetto ideato e sviluppato per la **NASA Space Apps Challenge 2026** da:

* **Marco Patrizi**
* **Eleonora Maria Piroli**
* **Marcello Mormone**
* **Emilie Mus**
* **Carlo Maria Raimondo**
* **Alessandro Ricchiuti**

---

## 📚 Fonti dati

Dataset e pubblicazioni dal portale pubblico **NASA Physical Science Informatics (PSI)** (*Microgravity Combustion Science*). I valori tabellari sono i dati delle figure degli articoli indicati nel file ReadMe NASA (DOI riportati nel catalogo dell'app).
