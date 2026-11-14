"""Indicizzazione offline dei report PDF in ChromaDB (senza LangChain).

* Embeddings: all-MiniLM-L6-v2 via ONNX (chromadb DefaultEmbeddingFunction) - stessi vettori del
  modello sentence-transformers, ma senza torch: leggero e compatibile con Streamlit Community Cloud.
* Chunk per pagina (mai a cavallo di due pagine) da ~1000 caratteri con overlap 200.
* Metadati: experiment_tag (PSI-XXX), file_name, page_number, doc_type, source, page.

Uso:
  python build_vector_db.py --rebuild     # ricrea la collezione dai PDF in ./dati_nasa_raw/report_pdf
  python build_processed.py               # poi rigenera pagine, sezioni e catalogo documenti
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from flame.config import CHROMA_COLLECTION, RAW_PDF_SRC, VECTOR_DB_DIR, experiment_tag

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200


def read_pdf_pages(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Spezza preferibilmente su paragrafo / riga / frase / spazio, con sovrapposizione."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > size // 2:
                    end = start + cut + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def doc_type(fname: str) -> str:
    low = fname.lower()
    if "dissertation" in low or "thesis" in low:
        return "Tesi di dottorato"
    if "presentation" in low:
        return "Presentazione"
    return "Report tecnico" if "report" in low else "Documento"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(RAW_PDF_SRC))
    ap.add_argument("--db-dir", default=str(VECTOR_DB_DIR))
    ap.add_argument("--rebuild", action="store_true", help="elimina e ricrea la collezione esistente")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Nessun PDF trovato in {pdf_dir}")
        return 1

    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    client = chromadb.PersistentClient(path=args.db_dir)
    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION in existing:
        if not args.rebuild:
            print(f"La collezione '{CHROMA_COLLECTION}' esiste gia' in {args.db_dir}. Usa --rebuild per ricrearla.")
            return 1
        client.delete_collection(CHROMA_COLLECTION)
    col = client.create_collection(CHROMA_COLLECTION, embedding_function=DefaultEmbeddingFunction())

    ids, docs, metas = [], [], []
    for pdf in pdfs:
        pages = read_pdf_pages(pdf)
        print(f"{pdf.name}: {len(pages)} pagine")
        for pno, text in enumerate(pages):
            for ci, chunk in enumerate(split_text(text)):
                ids.append(f"{re.sub(r'[^A-Za-z0-9]+', '_', pdf.stem)}_p{pno + 1}_c{ci}")
                docs.append(chunk)
                metas.append({"source": str(Path("dati_nasa_raw") / "report_pdf" / pdf.name), "page": pno,
                              "page_label": str(pno + 1), "total_pages": len(pages), "file_name": pdf.name,
                              "experiment_tag": experiment_tag(pdf.name), "page_number": pno + 1,
                              "doc_type": doc_type(pdf.name), "title": "", "author": ""})
    for i in range(0, len(ids), 128):
        col.add(ids=ids[i:i + 128], documents=docs[i:i + 128], metadatas=metas[i:i + 128])
    print(f"Indicizzati {len(ids)} chunk in {args.db_dir}. Ora esegui: python build_processed.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
