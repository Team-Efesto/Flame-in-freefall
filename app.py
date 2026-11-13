import os
import json
import pandas as pd
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Flame in Freefall - NASA PSI", page_icon="🔥", layout="wide")

st.title("🔥 Flame in Freefall: NASA Microgravity Fire Safety")
st.write("Esplora i dataset sperimentali NASA PSI e interroga le pubblicazioni scientifiche tramite IA.")

# 1. Caricamento Dataset JSON (Dati Sperimentali)
@st.cache_data
def load_json_dataset():
    json_path = "./data/neural_network_dataset.json"
    if not os.path.exists(json_path):
        return pd.DataFrame(), {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data.get("training_data", [])), data.get("dataset_metadata", {})

df_dati, metadati_json = load_json_dataset()

# 2. Barra Laterale: Filtro per Esperimento (Collega Dati JSON e Report PDF)
st.sidebar.header("🔎 Filtro Esperimento")
tag_disponibili = ["Tutti gli esperimenti"]
if not df_dati.empty and "experiment_tag" in df_dati.columns:
    tags = sorted([t for t in df_dati["experiment_tag"].dropna().unique() if t != "UNKNOWN"])
    tag_disponibili.extend(tags)

selected_tag = st.sidebar.selectbox("Seleziona codice NASA PSI:", tag_disponibili)

if selected_tag != "Tutti gli esperimenti" and not df_dati.empty:
    df_filtrato = df_dati[df_dati["experiment_tag"] == selected_tag]
else:
    df_filtrato = df_dati

with st.expander(f"📊 Dati Sperimentali Estratti ({len(df_filtrato)} record visualizzati)", expanded=False):
    if not df_filtrato.empty:
        st.dataframe(df_filtrato, use_container_width=True)
    else:
        st.info("Nessun dato tabellare trovato. Esegui prima build_json.py.")

st.divider()

# 3. Inizializza Claude Haiku e Database Vettoriale
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
anthropic_workspace_id = st.secrets["ANTHROPIC_WORKSPACE_ID"]
llm = ChatAnthropic(
    model="claude-haiku-5-5",
    api_key=anthropic_api_key,
    default_headers={"anthropic-workspace-id": anthropic_workspace_id}
)

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

vector_store = load_vector_store()

# 4. Configura il Prompt RAG con Citazione Obbligatoria delle Fonti
prompt_template = ChatPromptTemplate.from_template("""
Sei un assistente esperto in sicurezza antincendio in microgravità per la NASA.
Rispondi alla domanda usando esclusivamente i documenti forniti nel contesto.
REGOLA FONDAMENTALE: Ogni volta che riporti un dato, un risultato sperimentale o un'affermazione, cita tra parentesi il codice esperimento, il nome del documento e la pagina da cui l'hai tratto (es. [Fonte: PSI-117 | psi-117_report.pdf, Pag. 4]).
Se l'informazione non è presente nel contesto, dichiaralo chiaramente senza inventare.

Contesto:
{context}

Domanda: {input}
""")

# Catena LCEL nativa (non richiede langchain.chains)
chain = prompt_template | llm | StrOutputParser()

search_kwargs = {
    "k": 15,         # Passa a Claude 15 blocchi invece di 4
    "fetch_k": 40    # Analizza i 40 migliori e ne sceglie 15 diversi tra loro
}
if selected_tag != "Tutti gli esperimenti":
    search_kwargs["filter"] = {"experiment_tag": selected_tag}

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs=search_kwargs
)


def format_docs(docs):
    formatted = []
    for doc in docs:
        tag = doc.metadata.get("experiment_tag", "N/D")
        nome = doc.metadata.get("file_name", doc.metadata.get("source", "Sconosciuta"))
        pagina = doc.metadata.get("page_number", "N/D")
        formatted.append(f"Esperimento: {tag} | Documento: {nome} | Pagina: {pagina}\nContenuto: {doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# 5. Interfaccia Chat
st.subheader("🧠 Assistente IA sui Report NASA")
user_query = st.chat_input("Es: Qual è la velocità di propagazione della fiamma nell'esperimento PSI-117?")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analisi dei report NASA in corso..."):
            # Recupera i documenti pertinenti da ChromaDB
            docs_trovati = retriever.invoke(user_query)
            contesto_formattato = format_docs(docs_trovati)
            
            # Genera la risposta con Claude
            risposta = chain.invoke({"context": contesto_formattato, "input": user_query})
            st.write(risposta)
            
            with st.expander("📚 Mostra le fonti originali consultate"):
                if not docs_trovati:
                    st.warning("Nessun documento pertinente trovato per questo filtro.")
                for i, doc in enumerate(docs_trovati):
                    tag = doc.metadata.get("experiment_tag", "N/D")
                    nome = doc.metadata.get("file_name", doc.metadata.get("source", "Sconosciuta"))
                    pagina = doc.metadata.get("page_number", "N/D")
                    st.markdown(f"**Fonte {i+1}:** `{tag}` — `{nome}` (Pagina {pagina})")
                    st.caption(doc.page_content[:350] + "...")