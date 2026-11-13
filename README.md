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

```text
Flame-in-freefall/
├── .streamlit/
│   └── secrets.toml                # Chiavi API locali (escluso da Git)
├── data/
│   └── neural_network_dataset.json # Dataset unificato estratti dai CSV/XLSX
├── vector_db/                      # Database vettoriale ChromaDB pre-indicizzato
├── app.py                          # Applicazione principale Streamlit (Dashboard + RAG)
├── build_json.py                   # Script offline per pulizia e merge delle tabelle NASA
├── build_vector_db.py              # Script offline per indicizzazione PDF e metadati
├── requirements.txt                # Dipendenze Python del progetto
└── README.md                       # Documentazione del progetto
```

---

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

## 📚 Fonti Dati

I dataset e le pubblicazioni scientifiche utilizzati provengono dal portale pubblico **NASA Physical Science Informatics (PSI)** (*Microgravity Combustion Science*).