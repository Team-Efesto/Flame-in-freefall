"""Retrieval ibrido sui report PDF: semantico (ChromaDB, MiniLM-L6-v2 via ONNX) + parole chiave (BM25),
fusi con Reciprocal Rank Fusion ed espansi a livello di pagina (unita' di citazione).
"""
from __future__ import annotations

import math
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHROMA_COLLECTION, VECTOR_DB_DIR

_STOP = set("""a an the of and or in on at to for from by with without is are was were be been being as that this these those
it its into than then there their which who whom what when where how why do does did not no can could should would may might
il lo la i gli le un una di da in con su per tra fra e o che chi cosa come quale quali del della dei delle nel nella al alla
è sono sia""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avg = float(self.lens.mean()) if len(self.lens) else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tfs))
        for i, tf in enumerate(self.tfs):
            denom_base = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + denom_base)
            out[i] = s
        return out


@dataclass
class Hit:
    file_name: str
    page_number: int
    experiment_tag: str
    section: str
    doc_type: str
    score: float
    text: str
    matched_by: str

    @property
    def citation(self) -> str:
        return f"[Fonte: {self.experiment_tag} | {self.file_name}, p. {self.page_number}]"


def _open_collection(vector_dir: Path):
    """Apre Chroma su una copia temporanea: l'app non modifica mai i file versionati del repo."""
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    tmp = Path(tempfile.mkdtemp(prefix="flame_vdb_"))
    shutil.copytree(vector_dir, tmp / "db")
    client = chromadb.PersistentClient(path=str(tmp / "db"))
    return client.get_collection(CHROMA_COLLECTION, embedding_function=DefaultEmbeddingFunction())


class ReportIndex:
    def __init__(self, pages: pd.DataFrame, vector_dir: Path = VECTOR_DB_DIR, use_vectors: bool = True):
        self.pages = pages.reset_index(drop=True)
        self.key_to_row = {(r.file_name, int(r.page_number)): i for i, r in enumerate(self.pages.itertuples())}
        self.bm25 = BM25([f"{r.title} {r.section} {r.text}" for r in self.pages.itertuples()])
        self.collection = None
        self.vector_error = None
        if use_vectors:
            try:
                self.collection = _open_collection(vector_dir)
                self.collection.query(query_texts=["warm-up"], n_results=1)  # scarica/carica il modello ONNX subito
            except Exception as e:  # il retrieval resta funzionante con il solo BM25
                self.vector_error = str(e)

    def _mask(self, experiment_tag: str | None, file_name: str | None) -> np.ndarray:
        m = np.ones(len(self.pages), dtype=bool)
        if experiment_tag:
            m &= (self.pages.experiment_tag == experiment_tag).to_numpy()
        if file_name:
            m &= (self.pages.file_name == file_name).to_numpy()
        return m

    def search(self, query: str, experiment_tag: str | None = None, file_name: str | None = None,
               top_k: int = 6, max_chars: int = 2500) -> list[Hit]:
        mask = self._mask(experiment_tag, file_name)
        ranks: dict[int, float] = {}
        matched: dict[int, set] = {}
        best_chunk: dict[int, str] = {}

        # 1) parole chiave (BM25 a livello di pagina)
        sc = self.bm25.scores(query)
        sc[~mask] = 0
        for rank, i in enumerate([i for i in np.argsort(-sc)[:30] if sc[i] > 0]):
            ranks[i] = ranks.get(i, 0) + 1 / (60 + rank)
            matched.setdefault(i, set()).add("parole chiave")

        # 2) semantico (chunk Chroma -> pagina)
        if self.collection is not None:
            where = {}
            conds = [{"experiment_tag": experiment_tag}] if experiment_tag else []
            if file_name:
                conds.append({"file_name": file_name})
            if len(conds) == 1:
                where = conds[0]
            elif conds:
                where = {"$and": conds}
            try:
                res = self.collection.query(query_texts=[query], n_results=40, where=where or None,
                                            include=["metadatas", "documents"])
                seen = []
                for meta, doc in zip(res["metadatas"][0], res["documents"][0]):
                    i = self.key_to_row.get((meta.get("file_name"), int(meta.get("page_number", 0))))
                    if i is None or i in seen:
                        continue
                    seen.append(i)
                    best_chunk.setdefault(i, doc)
                for rank, i in enumerate(seen[:30]):
                    ranks[i] = ranks.get(i, 0) + 1 / (60 + rank)
                    matched.setdefault(i, set()).add("semantico")
            except Exception as e:
                self.vector_error = str(e)

        hits = []
        for i, score in sorted(ranks.items(), key=lambda kv: -kv[1])[:top_k]:
            r = self.pages.iloc[i]
            text = r.text
            if len(text) > max_chars:  # finestra centrata sul chunk piu' pertinente
                anchor = best_chunk.get(i, "")[:80]
                pos = max(0, text.find(anchor)) if anchor else 0
                start = max(0, min(pos - max_chars // 4, len(text) - max_chars))
                text = ("…" if start else "") + text[start:start + max_chars] + "…"
            hits.append(Hit(r.file_name, int(r.page_number), r.experiment_tag, r.section, r.doc_type,
                            round(float(score), 5), text, "+".join(sorted(matched[i]))))
        return hits

    def read_pages(self, file_name: str, start: int, end: int, max_pages: int = 8) -> pd.DataFrame:
        end = min(end, start + max_pages - 1)
        g = self.pages[(self.pages.file_name == file_name) & self.pages.page_number.between(start, end)]
        return g.sort_values("page_number")

    def resolve_file(self, name: str) -> str | None:
        """Accetta nomi file parziali (il modello a volte abbrevia)."""
        files = self.pages.file_name.unique().tolist()
        if name in files:
            return name
        low = name.lower()
        cands = [f for f in files if low in f.lower()]
        return cands[0] if len(cands) == 1 else None
