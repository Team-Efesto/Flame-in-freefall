import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_nasa_vector_db(pdf_folder_path, persist_dir):
    print(f"📚 Lettura dei PDF dalla cartella: {pdf_folder_path}")
    
    if not os.path.exists(pdf_folder_path):
        print(f"❌ Errore: La cartella {pdf_folder_path} non esiste.")
        return

    # 1. Carica tutti i PDF nella cartella
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()
    
    if not documents:
        print("❌ Nessun file PDF trovato nella cartella.")
        return
        
    print(f"✅ Pagine PDF estratte: {len(documents)}")

    # 2. Divide i testi in blocchi (chunk) gestibili dall'IA
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Testo suddiviso in {len(chunks)} blocchi.")

    # 3. Genera gli embeddings (stesso identico modello usato in app.py)
    print("🧠 Generazione dei vettori in corso (al primo avvio scaricherà il modello)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Salva il database vettoriale su disco
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"🚀 Database vettoriale salvato con successo in: {persist_dir}")

if __name__ == "__main__":
    CARTELLA_PDF = r"./dati_nasa_raw/report_pdf"
    CARTELLA_VECTOR_DB = r"./vector_db"

    create_nasa_vector_db(CARTELLA_PDF, CARTELLA_VECTOR_DB)