<<<<<<< HEAD
# 🔥 Flame in Freefall — NASA Microgravity Fire Safety Dashboard

**NASA Space Apps Challenge 2026**  
Una piattaforma web basata su Intelligenza Artificiale per esplorare, correlare e interrogare i dataset sperimentali e le pubblicazioni scientifiche del database **NASA Physical Science Informatics (PSI)** sulla combustione e sicurezza antincendio in microgravità.

---

## 🚀 Panoramica del Progetto

Lo studio del comportamento delle fiamme in condizioni di microgravità è cruciale per la sicurezza delle missioni spaziali di lunga durata. Tuttavia, i dati sperimentali (telemetria, sensori termici, velocità di propagazione) e i risultati scientifici (tesi, report tecnici, presentazioni) sono spesso frammentati in migliaia di file eterogenei.

**Flame in Freefall** risolve questo problema unendo l'analisi dei dati tabellari a un'architettura **RAG (Retrieval-Augmented Generation)**:
* **Correlazione Automatica (`PSI-XXX`):** Associa univocamente i record numerici (CSV/Excel) alle relative pubblicazioni scientifiche (PDF) tramite il codice identificativo NASA PSI.
* **Assistente AI con Citazione delle Fonti:** Permette a ricercatori e utenti di porre domande in linguaggio naturale sui report NASA, ricevendo risposte basate esclusivamente sui documenti reali con indicazione esatta del file e del numero di pagina.
* **Pipeline Dati Dinamica:** Estrae e standardizza le variabili principali preservando al contempo tutte le colonne dei sensori aggiuntivi senza alterare i valori mancanti.

---

## ✨ Funzionalità Principali

* **Filtro per Esperimento NASA PSI:** Selezione rapida tramite barra laterale per isolare contemporaneamente i dati sperimentali e i documenti PDF di uno specifico studio (es. `PSI-117`).
* **Esplorazione Dataset Tabellari:** Visualizzazione interattiva dei dati estratti dai file grezzi NASA e convertiti in formato JSON strutturato (`neural_network_dataset.json`), pronto anche per l'addestramento di reti neurali.
* **Ricerca Semantica Diversificata (MMR):** Utilizzo di *Maximal Marginal Relevance* su database vettoriale ChromaDB per estrarre informazioni rilevanti da tutto il corpo dei documenti tecnici, evitando ridondanze.
* **Trasparenza e Verificabilità:** Ogni risposta dell'assistente include citazioni puntuali nel testo e un pannello espandibile per consultare gli estratti originali dei report.

---

## 🛠️ Stack Tecnologico

* **Frontend & Dashboard:** [Streamlit](https://streamlit.io/)
* **Orchestrazione AI:** [LangChain](https://www.langchain.com/) (LCEL)
* **Large Language Model (LLM):** Anthropic **Claude Haiku** (`langchain-anthropic`)
* **Embeddings & Vector Store:** HuggingFace (`all-MiniLM-L6-v2`) + [ChromaDB](https://www.trychroma.com/)
* **Data Processing:** Python, Pandas, OpenPyXL, PyPDF

---

## 📂 Struttura della Repository
=======
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
| 📊 **Dashboard dati** | Esploratore di serie con scala log, tabella dei punti e tabella originale NASA, download CSV/XLSX. |
| 📚 **Report scientifici** | Sintesi per esperimento, sommario e risultati chiave (con pagine) per ogni PDF, indice delle sezioni, ricerca ibrida, lettore di pagine. |
| 🛡️ **Norme di sicurezza** | Limiti numerici antincendio con testo originale e pagina, requisiti NASA-STD-3001, verifica automatica delle serie di simulazione/modello, verifica di un file di simulazione caricato (CSV/XLSX), ricerca e lettura delle norme. |
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

data_nasa_raw/safety_pdf/ ──► build_safety_db.py ──► vector_db_safety/ (ChromaDB, collezione "safety")
                                    │            └─► data/safety/*.parquet (pagine, limiti, requisiti)
                                    ▼
                         flame/safety.py: SafetyLimits ──► confronto simulazioni ↔ norme antincendio
```

* **Dati tabellari** — le tabelle NASA sono "figure data" in formato largo (più serie x/y affiancate). `flame/tidy.py` le converte in un formato lungo (una riga per punto) con serie, tipo (experiment/model/simulation), combustibile, pressione, d0, grandezze canoniche e unità, mantenendo la tracciabilità alla colonna originale.
* **Query engine** — Claude usa lo strumento `query_data` (SQL DuckDB) sull'intero dataset. Sicurezza: una sola `SELECT`, parole chiave pericolose bloccate, `enable_external_access=false` + `lock_configuration=true` (nessun accesso a file o rete), limite righe. Al modello arrivano solo anteprime e statistiche; i risultati completi restano all'app per grafici e download.
* **RAG sui report** — retrieval ibrido (embeddings `all-MiniLM-L6-v2` + BM25, fusione RRF) a livello di **pagina** con sezione/capitolo; lettura di intervalli di pagine; sommari strutturati per documento e per esperimento generati offline (nessun costo a runtime).
* **Norme di sicurezza antincendio** — i PDF NASA di sicurezza (NASA-STD-3001 / OCHMO-TB-008, *Fire Safety Applications for Spacecraft*, ECLS Fire Safety, ecc.) hanno un database **separato** (`vector_db_safety/`, stessa vettorizzazione dei report PSI, `experiment_tag = "SAFETY"`) e tabelle strutturate in `data/safety/`: limiti numerici in unità canoniche con citazione verificata (`safety_limits`), requisiti `[V2 xxxx]` (`safety_requirements`). `flame/safety.py` confronta i risultati delle simulazioni con questi limiti (vedi sotto).
* **Strumenti dell'agente** — `query_data`, `create_chart`, `search_reports`, `read_report_pages`, `get_document_overview`, `get_raw_table`, `suggest_explorer` e, se il database delle norme è presente, `search_safety_norms` e `check_safety_compliance` (il catalogo dei limiti è nel prompt di sistema, in cache).

---

## 🛠️ Stack

Streamlit · Anthropic Python SDK (Claude Haiku 5.5 con tool use e prompt caching) · DuckDB · Pandas/PyArrow (Parquet) · Plotly · ChromaDB (embeddings ONNX `all-MiniLM-L6-v2`, senza torch) · OpenPyXL · PyPDF

---

## 📂 Struttura
>>>>>>> dev

```text
Flame-in-freefall/
├── .streamlit/
<<<<<<< HEAD
│   └── secrets.toml                # Chiavi API locali (escluso da Git)
├── data/
│   └── neural_network_dataset.json # Dataset unificato estratti dai CSV/XLSX
├── vector_db/                      # Database vettoriale ChromaDB pre-indicizzato
├── app.py                          # Applicazione principale Streamlit (Dashboard + RAG)
├── build_json.py                   # Script offline per pulizia e merge delle tabelle NASA
├── build_vector_db.py              # Script offline per indicizzazione PDF e metadati
├── requirements.txt                # Dipendenze Python del progetto
└── README.md                       # Documentazione del progetto
=======
│   ├── config.toml                  # Tema (versionato)
│   └── secrets.toml                 # Chiavi API locali (escluso da Git)
├── data/
│   ├── neural_network_dataset.json  # Dataset unificato dai CSV/XLSX NASA
│   ├── processed/                   # Artefatti interrogabili (Parquet) + sommari
│   └── safety/                      # Norme antincendio: pagine, limiti, requisiti (Parquet/CSV)
├── flame/                           # Pacchetto applicativo
│   ├── agent.py                     # Agente Claude con tool use
│   ├── charts.py                    # Grafici Plotly (palette accessibile, tema scuro)
│   ├── config.py                    # Percorsi e costanti
│   ├── data_store.py                # Parquet + SQL read-only (DuckDB)
│   ├── export.py                    # Export CSV / Excel con foglio metadati
│   ├── retrieval.py                 # Retrieval ibrido sui report
│   ├── safety.py                    # Confronto simulazioni ↔ limiti di sicurezza antincendio
│   └── tidy.py                      # Formato largo → formato lungo
├── vector_db/                       # ChromaDB pre-indicizzato (report PSI)
├── vector_db_safety/                # ChromaDB delle norme di sicurezza antincendio
├── app.py                           # Applicazione Streamlit
├── build_json.py                    # (offline) CSV/XLSX → JSON
├── build_vector_db.py               # (offline) PDF → ChromaDB
├── build_processed.py               # (offline) JSON/vector DB → data/processed
├── build_safety_db.py               # (offline) PDF di sicurezza → vector_db_safety + data/safety
├── ARCHITECTURE_PROGRESS.md         # Stato dei lavori e scelte architetturali
└── requirements.txt
>>>>>>> dev
```

---

<<<<<<< HEAD
## ⚙️ Installazione e Avvio in Locale

### 1. Clona la repository e crea l'ambiente virtuale
```bash
git clone [https://github.com/tuo-username/Flame-in-freefall.git](https://github.com/tuo-username/Flame-in-freefall.git)
cd Flame-in-freefall

python -m venv .venv
# Su Windows:
.venv\Scripts\activate
# Su macOS/Linux:
source .venv/bin/activate
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Configura le chiavi API
Crea il file `.streamlit/secrets.toml` nella directory principale del progetto e inserisci la tua chiave API di Anthropic:
```toml
ANTHROPIC_API_KEY = "sk-ant-api03-la-tua-chiave-qui"
```

### 4. (Opzionale) Rigenerazione dei Database dai file grezzi NASA
Se desideri aggiornare i dataset partendo da nuovi file scaricati dal portale NASA PSI, inseriscili nelle cartelle `./dati_nasa_raw/tabelle` (CSV/XLSX) e `./dati_nasa_raw/report_pdf` (PDF), assicurandoti che i nomi contengano il codice `PSI-XXX`, ed esegui:
```bash
python build_json.py
python build_vector_db.py
```

### 5. Avvia la Dashboard
```bash
streamlit run app.py
```
=======
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

### Database delle norme di sicurezza antincendio
I PDF vanno in `./data_nasa_raw/safety_pdf` (cartella non versionata). Per rigenerare:
```bash
python build_safety_db.py --rebuild
```
Ogni limite in `LIMIT_RULES` (dentro `build_safety_db.py`) ha un'**ancora testuale**: lo script trova file e pagina nel PDF e si interrompe se il testo non c'è, così ogni valore resta tracciabile. Per aggiungere un limite basta aggiungere una regola.

| ID | Grandezza | Limite | Tipo | Fonte |
|---|---|---|---|---|
| SAF-001 | `burn_length` | ≤ 15 cm (test di infiammabilità verso l'alto) | requisito | Fire Safety Applications, p. 4 |
| SAF-002 | `air_velocity` | 8–20 cm/s (ventilazione nominale) | riferimento progetto | idem, p. 7 |
| SAF-003 | `air_velocity` | zona di rischio 8–20 cm/s (massima propagazione in µg) | soglia sperimentale | idem, p. 10 |
| SAF-004 | `oxygen_concentration` | zona di rischio ≥ 15 % (la carta brucia in µg) | soglia sperimentale | idem, p. 10 |
| SAF-005 | `oxygen_concentration` | ≤ 21 % (aria a livello del mare)¹ | riferimento progetto | idem, p. 9 |
| SAF-006* | `oxygen_concentration` | ≤ 6 % (volumi inertizzati non abitati) | soglia sperimentale | idem, p. 12 |
| SAF-007 | `material_separation` | > 1.3 cm (propagazione osservata su Skylab) | soglia sperimentale | idem, p. 7 |
| SAF-008* | `temperature` | ≤ 1366 K / 2000 °F (coperte termiche multistrato) | capacità dispositivo | Multilayered Fire Protection, p. 1 |
| SAF-009* | `ammonia_concentration` | ≤ 30 000 ppm (maschera d'emergenza) | capacità dispositivo | ECLS Fire Safety, p. 25 |
| SAF-010* | `exposure_time` | ≤ 15 min (durata della maschera) | capacità dispositivo | idem, p. 25 |

¹ Il documento cita "sea-level air" senza riportare il numero (`value_in_text = False`).
\* Ambito ristretto: esclusi dai controlli automatici, si attivano con `include_all=True` o `limit_ids=[...]`.

I documenti **non** riportano valori numerici per i gas tossici (CO, HCN, HCl): rimandano alle tabelle SMAC (JSC-20584), che non fanno parte del corpus.

### Confronto delle simulazioni con le norme
```python
from flame.safety import SafetyLimits
lim = SafetyLimits()

lim.check([1800, 2600], "temperature", "K", include_all=True)   # valori con grandezza e unità
lim.check_dataframe(sim_df)            # colonne riconosciute dal nome: temperature_K, "O2 [%]", x_o2, air_velocity_m_s, burn_length_mm…
lim.check_dataframe(sim_df, mapping={"T": ("temperature", "K")})   # mapping esplicito
lim.check_store(DataStore(), kinds=("simulation",))                 # serie già presenti nell'app
```
Le unità vengono convertite automaticamente (m/s → cm/s, °F/°C → K, frazione molare → %, atm/psi → kPa…). Per ogni coppia grandezza/limite il risultato riporta:
* `esito`: *conforme* / *non conforme* per le zone conformi, *in zona di rischio* / *fuori zona di rischio* per le zone di rischio;
* `n_violations` e `worst_value`;
* `margin`: distanza dal limite, positiva se dal lato sicuro;
* `citation`: file e pagina della fonte.

Le grandezze solo affini (es. `flame_length` rispetto al limite di `burn_length`) si confrontano soltanto con `allow_proxy=True` e sono marcate "indicativo (proxy)".

### Deploy su Streamlit Community Cloud
Imposta `ANTHROPIC_API_KEY` (e opzionalmente `ANTHROPIC_WORKSPACE_ID`) nei *Secrets* dell'app. Al primo avvio viene scaricato una sola volta il modello di embedding ONNX (~80 MB).
>>>>>>> dev

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

<<<<<<< HEAD
## 📚 Fonti Dati

I dataset e le pubblicazioni scientifiche utilizzati provengono dal portale pubblico **NASA Physical Science Informatics (PSI)** (*Microgravity Combustion Science*).
=======
## 📚 Fonti dati

Dataset e pubblicazioni dal portale pubblico **NASA Physical Science Informatics (PSI)** (*Microgravity Combustion Science*). I valori tabellari sono i dati delle figure degli articoli indicati nel file ReadMe NASA (DOI riportati nel catalogo dell'app).
>>>>>>> dev
