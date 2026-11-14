"""Percorsi e costanti condivise tra app e script di build."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_TABLES_DIR = PROCESSED_DIR / "raw_tables"
VECTOR_DB_DIR = ROOT / "vector_db"
LEGACY_JSON = DATA_DIR / "neural_network_dataset.json"

# Cartelle grezze locali (non versionate, usate solo dagli script di build)
RAW_TABLES_SRC = ROOT / "dati_nasa_raw" / "tabelle"
RAW_PDF_SRC = ROOT / "dati_nasa_raw" / "report_pdf"

CHROMA_COLLECTION = "langchain"  # nome storico della collezione, mantenuto per compatibilita'
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # stesso modello, servito via ONNX da chromadb

DEFAULT_MODEL = "claude-haiku-5-5"

TAG_RE = re.compile(r"(psi-\d+)", re.IGNORECASE)


def experiment_tag(name: str) -> str:
    m = TAG_RE.search(name or "")
    return m.group(1).upper() if m else "UNKNOWN"
