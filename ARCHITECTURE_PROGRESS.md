# ARCHITECTURE_PROGRESS

Ultimo aggiornamento: 2026-11-14 — **Stato: prima iterazione completa e testata.** Non ancora committata.

## Scoperte iniziali (vincolano le scelte)
- Tutti i dati attuali (10 tabelle CSV/XLSX + 5 PDF, 322 pagine) appartengono a **PSI-117**: combustione di gocce isolate, cool flames, estinzione. **Non esistono** dati di velocità di propagazione della fiamma vs O2 né altri esperimenti: l'agente lo dichiara esplicitamente.
- Le tabelle sono "figure data" in formato largo: più serie x/y affiancate, colonne-etichetta (`EXP`, `Model`, `Simulation`, `0.5 atm`…) e separatori vuoti. Il `column_mapping` di `build_json.py` (O2_Concentration_pct…) non corrisponde a nessuna colonna reale.
- Il vector DB committato **non aveva** i metadati `experiment_tag/file_name/page_number`: il filtro per esperimento restituiva zero risultati e le citazioni mostravano "N/D". Ora corretto (patch non distruttiva su 588 chunk).
- `requirements.txt` era in UTF-16 senza versioni (rischio di install fallita su Cloud) e includeva torch/LangChain inutilizzati.
- `.gitignore` escludeva tutta `.streamlit/`, quindi `config.toml` (tema) non era mai stato deployato. Ora è escluso solo `secrets.toml`.
- `dati_nasa_raw/` non è presente: tutti gli artefatti sono stati ricostruiti dal JSON e dal contenuto di Chroma (nessuna rigenerazione dai dati grezzi).

## Scelte architetturali
| Area | Scelta | Motivo |
|---|---|---|
| Formato dati | Parquet in `data/processed/` + DuckDB in memoria | Interrogabile con SQL, leggero (884 KB), nessun server; il JSON resta come sorgente storica |
| Reshape | `flame/tidy.py` generico (nessuna regola per file): blocchi x/y, contesto da colonne-etichetta, doppio separatore azzera il contesto, grandezze canoniche + unità | 67 serie / 15.797 punti da 10 file; tracciabilità a `x_column`/`y_column` |
| Tipo serie | `kind` = experiment/model/simulation/unspecified; `kind_inferred` se dedotto | Distinguere misure da simulazioni senza inventare |
| LLM | Anthropic SDK diretto (non LangChain), `claude-haiku-5-5` (configurabile via secrets) | Tool use nativo, prompt caching sul prompt di sistema + catalogo |
| Tool | `query_data`, `create_chart`, `search_reports`, `read_report_pages`, `get_document_overview`, `get_raw_table` | Accesso completo ai dati; al modello solo anteprime (≤40 righe + statistiche) |
| Sicurezza SQL | 1 SELECT/WITH, blacklist parole chiave, `enable_external_access=false`, `lock_configuration=true`, LIMIT | SQL generato dal modello eseguito in sandbox |
| RAG | Pagina come unità; BM25 + Chroma (RRF); sezioni/capitoli rilevati; sommari LLM offline per doc ed esperimento | Domande di sintesi senza leggere 300 pagine; citazioni precise |
| Embeddings | `all-MiniLM-L6-v2` via ONNX (chromadb) — verificato coseno = 1.0 con i vettori esistenti | Elimina torch/sentence-transformers (RAM Cloud) |
| Vector DB a runtime | Copia in tempdir all'avvio | L'app non sporca più i file versionati |
| Grafici | Plotly, palette categoriale accessibile (dataviz skill, variante scura), max 8 colori → altrimenti errore al modello che usa facet/filtri; un solo asse y | Leggibilità e accessibilità |

## Test eseguiti
- AppTest headless: tutte le 37 combinazioni dataset × grandezza, toggle log, filtro, ricerca e lettore report, chat end-to-end con grafico/tabella/download: nessuna eccezione.
- Agente su 5 tipologie di domanda (quantitativa+grafico, estrazione tabella, sintesi, teorica, dati inesistenti) + follow-up: strumenti corretti, citazioni presenti, cache del prompt attiva.
- SQL guard: DROP, multi-statement, `read_csv`, `glob`, accesso diretto a file → bloccati.
- `build_vector_db.py` testato su PDF sintetico in DB temporaneo (guardia `--rebuild` verificata).
- `pip install --dry-run` dei requirements: 106 pacchetti, nessun conflitto, niente torch/LangChain.

## Passi rimanenti / idee
- [ ] Commit (file nuovi: `flame/`, `data/processed/`, `build_processed.py`, `.streamlit/config.toml`, `vector_db/*` aggiornati).
- [ ] Verifica deploy reale su Streamlit Community Cloud (Python 3.11+).
- [ ] Immagini dei report: non estraibili senza i PDF originali; con `dati_nasa_raw/report_pdf` si può aggiungere estrazione figure (pypdf) in `build_processed.py`.
- [ ] `build_json.py`: il `column_mapping` non corrisponde ai file reali (innocuo); valutare rimozione.
- [ ] Se arriveranno nuovi esperimenti PSI: rieseguire la pipeline di build; il reshape è generico ma va controllato il catalogo serie.
- [ ] Eventuale streaming del testo della risposta (oggi: stato dei tool in tempo reale, testo a fine analisi).
