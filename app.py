import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.title("🧠 Assistente IA per la Sicurezza Antincendio")
st.write("Fai una domanda sulle pubblicazioni e i report della NASA.")

# 1. Inizializza l'IA (richiede una API key salvata nei segreti di Streamlit)
# Nota: su Streamlit Cloud le API key si salvano nella sezione "Secrets", non nel codice.
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# 2. Carica il database vettoriale dalla cartella del progetto (precedentemente pushato)
@st.cache_resource
def load_vector_store():
    # Punta alla cartella che hai generato offline e caricato su GitHub
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
    return vector_store

vector_store = load_vector_store()

# 3. Configura il sistema RAG (Retrieval)
prompt_template = ChatPromptTemplate.from_template("""
Sei un assistente esperto in sicurezza spaziale per la NASA. 
Usa il seguente contesto tratto dalle pubblicazioni scientifiche per rispondere alla domanda.
Se non conosci la risposta in base al contesto, dillo chiaramente.

Contesto:
{context}

Domanda: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Prende i 3 paragrafi più rilevanti
rag_chain = create_retrieval_chain(retriever, document_chain)

# 4. Interfaccia Chat
user_query = st.chat_input("Es: Come si comporta una fiamma con il 30% di ossigeno?")

if user_query:
    # Mostra la domanda dell'utente
    with st.chat_message("user"):
        st.write(user_query)
        
    # Elabora e mostra la risposta dell'IA
    with st.chat_message("assistant"):
        with st.spinner("Analisi dei report NASA in corso..."):
            response = rag_chain.invoke({"input": user_query})
            st.write(response["answer"])
            
            # Mostra da dove ha preso le informazioni (Fonti)
            with st.expander("Mostra le fonti (Report originali)"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Fonte {i+1}:** {doc.metadata.get('source', 'Sconosciuta')}")
                    st.caption(doc.page_content[:300] + "...")