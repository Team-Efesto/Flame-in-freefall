"""Build offline degli artefatti interrogabili usati dall'app (cartella data/processed/).

Sorgenti (in ordine di preferenza):
  * tabelle: ./dati_nasa_raw/tabelle (CSV/XLSX) se presente, altrimenti data/neural_network_dataset.json
  * report : ./dati_nasa_raw/report_pdf se presente (via build_vector_db.py), altrimenti i testi
             gia' indicizzati in ./vector_db (ChromaDB): nessun file grezzo e' necessario.

Output:
  data/processed/points.parquet     una riga per punto (x, y) di ogni serie, formato lungo
  data/processed/series.parquet     catalogo delle serie (variabili, unita', condizioni, range)
  data/processed/datasets.parquet   una riga per file/figura con metadati e DOI
  data/processed/raw_tables/*.parquet  tabelle originali (formato largo) per l'export fedele
  data/processed/doc_pages.parquet  testo completo per pagina, con sezione/capitolo
  data/processed/documents.parquet  catalogo report + indice sezioni + sommario strutturato
  data/processed/experiments.parquet sintesi per esperimento PSI-XXX

Uso:
  python build_processed.py                 # tabelle + documenti + metadati Chroma
  python build_processed.py --summaries     # genera anche i sommari con Claude (richiede API key)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from flame.config import (DEFAULT_MODEL, LEGACY_JSON, PROCESSED_DIR, RAW_PDF_SRC, RAW_TABLES_DIR, RAW_TABLES_SRC,
                          ROOT, VECTOR_DB_DIR, CHROMA_COLLECTION, experiment_tag)
from flame.tidy import dataset_name, parse_readme, publication, tidy_table

SUMMARY_CACHE = PROCESSED_DIR / "summaries.json"


# ---------------------------------------------------------------------------
# 1. Tabelle
# ---------------------------------------------------------------------------
def load_source_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if RAW_TABLES_SRC.exists():
        for p in sorted(list(RAW_TABLES_SRC.glob("*.csv")) + list(RAW_TABLES_SRC.glob("*.xlsx"))):
            if p.suffix.lower() == ".csv":
                try:
                    tables[p.name] = pd.read_csv(p, low_memory=False, encoding="utf-8")
                except UnicodeDecodeError:
                    tables[p.name] = pd.read_csv(p, low_memory=False, encoding="cp1252")
            else:
                tables[p.name] = pd.read_excel(p)
        print(f"Tabelle lette da {RAW_TABLES_SRC}: {len(tables)}")
        return tables
    data = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in data["training_data"]:
        by_file[rec["source_file"]].append(rec)
    for f, recs in by_file.items():
        cols = [c for c in recs[0].keys() if c not in ("source_file", "experiment_tag")]  # ordine originale
        tables[f] = pd.DataFrame([{c: r.get(c) for c in cols} for r in recs], columns=cols)
    print(f"Tabelle ricostruite da {LEGACY_JSON.name}: {len(tables)}")
    return tables


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = load_source_tables()
    figures = pd.concat([parse_readme(df, f) for f, df in tables.items() if "readme" in f.lower()]
                        or [pd.DataFrame()], ignore_index=True)
    fig_by_ds = {r["dataset"]: r for r in figures.to_dict("records")} if not figures.empty else {}

    RAW_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_series, all_points, ds_rows = [], [], []
    for f, df in tables.items():
        ds = dataset_name(f)
        raw = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        raw.astype(str).replace({"None": None, "nan": None}).to_parquet(RAW_TABLES_DIR / f"{ds}.parquet", index=False)
        is_readme = "readme" in f.lower()
        fig = fig_by_ds.get(ds, {})
        series, points = ([], pd.DataFrame()) if is_readme else tidy_table(f, df, fig)
        all_series += series
        if not points.empty:
            all_points.append(points)
        journal, year = publication(ds)
        ds_rows.append({
            "dataset": ds, "experiment_tag": experiment_tag(f), "source_file": f,
            "role": "readme" if is_readme else "figure_data",
            "journal": journal or None, "year": year,
            "figure": fig.get("figure"), "fuel": fig.get("fuel"), "d0_mm": fig.get("d0_mm"),
            "pressure_atm": fig.get("pressure_atm"), "x_o2": fig.get("x_o2"), "x_n2": fig.get("x_n2"),
            "reference_doi": fig.get("reference_doi"),
            "n_series": len(series), "n_points": int(sum(s["n_points"] for s in series)),
            "raw_rows": int(len(raw)), "raw_columns": int(raw.shape[1]),
        })
    points = pd.concat(all_points, ignore_index=True)
    series = pd.DataFrame(all_series)
    datasets = pd.DataFrame(ds_rows)
    datasets["year"] = datasets["year"].astype("Int64")
    points.to_parquet(PROCESSED_DIR / "points.parquet", index=False)
    series.to_parquet(PROCESSED_DIR / "series.parquet", index=False)
    datasets.to_parquet(PROCESSED_DIR / "datasets.parquet", index=False)
    print(f"Serie: {len(series)} | punti: {len(points)} | dataset: {len(datasets)}")
    return series, datasets


# ---------------------------------------------------------------------------
# 2. Documenti (pagine complete + sezioni)
# ---------------------------------------------------------------------------
def _merge_chunks(chunks: list[str]) -> str:
    """Ricompone il testo di una pagina dai chunk sovrapposti (overlap ~200 caratteri)."""
    out = chunks[0]
    for c in chunks[1:]:
        probe = c[:60]
        idx = out.rfind(probe, max(0, len(out) - 600)) if len(probe) >= 20 else -1
        out = out[:idx] + c if idx >= 0 else out + "\n" + c
    return out


def pages_from_chroma() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{VECTOR_DB_DIR / 'chroma.sqlite3'}?mode=ro", uri=True)
    rows = con.execute(
        "select e.seq_id, m.key, m.string_value, m.int_value from embeddings e "
        "join embedding_metadata m on m.id = e.id "
        "where m.key in ('source','page','page_label','title','author','chroma:document') order by e.seq_id"
    ).fetchall()
    con.close()
    recs: dict[int, dict] = collections.defaultdict(dict)
    for seq, k, sv, iv in rows:
        recs[seq][k] = sv if sv is not None else iv
    grouped: dict[tuple, list] = collections.defaultdict(list)
    meta: dict[tuple, dict] = {}
    for seq in sorted(recs):
        r = recs[seq]
        key = (re.split(r"[\\/]", r["source"])[-1], int(r["page"]))
        grouped[key].append(r["chroma:document"])
        meta.setdefault(key, r)
    out = []
    for (fname, page), chunks in grouped.items():
        m = meta[(fname, page)]
        out.append({"file_name": fname, "page_number": page + 1, "page_label": str(m.get("page_label") or page + 1),
                    "pdf_title": m.get("title") or "", "pdf_author": m.get("author") or "",
                    "text": _merge_chunks(chunks)})
    return pd.DataFrame(out).sort_values(["file_name", "page_number"]).reset_index(drop=True)


def pages_from_pdfs() -> pd.DataFrame:
    """Testo pagina per pagina direttamente dai PDF grezzi (se la cartella locale esiste)."""
    from pypdf import PdfReader
    out = []
    for pdf in sorted(RAW_PDF_SRC.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        info = reader.metadata or {}
        for i, page in enumerate(reader.pages):
            out.append({"file_name": pdf.name, "page_number": i + 1, "page_label": str(i + 1),
                        "pdf_title": str(info.get("/Title") or ""), "pdf_author": str(info.get("/Author") or ""),
                        "text": page.extract_text() or ""})
    return pd.DataFrame(out)


def doc_type(fname: str) -> str:
    low = fname.lower()
    if "dissertation" in low or "thesis" in low:
        return "Tesi di dottorato"
    if "presentation" in low:
        return "Presentazione"
    if "report" in low:
        return "Report tecnico"
    return "Documento"


def doc_title(fname: str, pdf_title: str) -> str:
    if pdf_title and pdf_title.strip() and pdf_title.strip().lower() not in {"powerpoint presentation", "slide 1"}:
        return pdf_title.strip()
    stem = re.sub(r"\.pdf$", "", fname, flags=re.IGNORECASE)
    stem = re.sub(r"^PSI-\d+_?(Presentations|Reports)?_?(Dissertation)?_?", "", stem, flags=re.IGNORECASE)
    return stem.replace("_", " ").strip()


CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+(\d+)\s*[:.\-]?\s*(.*)$")
SECTION_RE = re.compile(r"^\s*(\d+\.\d+)\s+([A-Z][A-Z0-9 ,\-:()/&'’]{4,})\s*$")
FRONT_RE = re.compile(r"^\s*(ABSTRACT|PREFACE|DEDICATION|ACKNOWLEDGE?MENTS|TABLE OF CONTENTS|LIST OF TABLES|"
                      r"LIST OF FIGURES|LIST OF ABBREVIATIONS|REFERENCES|BIBLIOGRAPHY|APPENDIX\s*[A-Z]?)\s*$")


def assign_sections(pages: pd.DataFrame) -> pd.DataFrame:
    sections = []
    for fname, g in pages.groupby("file_name", sort=False):
        is_deck = doc_type(fname) == "Presentazione"
        chapter, section = "", ""
        for _, r in g.iterrows():
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
            if is_deck:
                title = next((ln for ln in lines if len(ln) > 3 and not re.fullmatch(r"[\d\s./]+", ln)), "")
                sections.append(title[:120])
                continue
            is_toc = r.text.count(".....") >= 3  # pagine dell'indice / elenchi con puntini guida
            for li, ln in enumerate(lines[:40]):
                if "...." in ln:
                    continue
                m = CHAPTER_RE.match(ln)
                if m and ln.upper() == ln and not is_toc:
                    title = m.group(2).strip()
                    if not title and li + 1 < len(lines) and lines[li + 1].upper() == lines[li + 1]:
                        title = lines[li + 1]
                    title = re.sub(r"\s+", " ", title).title()
                    chapter, section = f"Capitolo {m.group(1)} {title}".strip(), ""
                    continue
                m = FRONT_RE.match(ln)
                if m:
                    chapter, section = m.group(1).title(), ""
                    continue
                m = SECTION_RE.match(ln)
                if m and chapter.startswith("Capitolo") and not is_toc:
                    section = f"{m.group(1)} {m.group(2).title()}"
            sections.append(" › ".join(x for x in (chapter, section) if x))
    pages = pages.copy()
    pages["section"] = sections
    return pages


def build_documents() -> pd.DataFrame:
    use_pdfs = RAW_PDF_SRC.exists() and any(RAW_PDF_SRC.glob("*.pdf"))
    pages = pages_from_pdfs() if use_pdfs else pages_from_chroma()
    print(f"Pagine lette da {'PDF grezzi' if use_pdfs else 'vector DB (chunk ricomposti)'}")
    pages["experiment_tag"] = pages.file_name.map(experiment_tag)
    pages["doc_type"] = pages.file_name.map(doc_type)
    titles = pages.groupby("file_name").pdf_title.agg(lambda s: next((t for t in s if t), ""))
    pages["title"] = [doc_title(f, titles[f]) for f in pages.file_name]
    pages = assign_sections(pages)
    pages["n_chars"] = pages.text.str.len()
    pages = pages[["experiment_tag", "file_name", "doc_type", "title", "page_number", "page_label",
                   "section", "n_chars", "text", "pdf_author"]]
    pages.to_parquet(PROCESSED_DIR / "doc_pages.parquet", index=False)
    print(f"Pagine documento: {len(pages)} in {pages.file_name.nunique()} PDF")
    return pages


def outline(g: pd.DataFrame) -> list[dict]:
    out, last = [], None
    for _, r in g.iterrows():
        if r.section and r.section != last:
            out.append({"section": r.section, "start_page": int(r.page_number)})
            last = r.section
    return out


# ---------------------------------------------------------------------------
# 3. Sommari strutturati (opzionali, generati una volta con Claude e messi in cache)
# ---------------------------------------------------------------------------
DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "object", "properties": {
            "finding": {"type": "string"}, "pages": {"type": "array", "items": {"type": "integer"}}},
            "required": ["finding", "pages"], "additionalProperties": False}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "experimental_conditions": {"type": "string"},
    },
    "required": ["summary", "key_findings", "topics", "experimental_conditions"],
    "additionalProperties": False,
}
EXP_SCHEMA = {
    "type": "object",
    "properties": {"overview": {"type": "string"}, "highlights": {"type": "array", "items": {"type": "string"}}},
    "required": ["overview", "highlights"], "additionalProperties": False,
}


def _client():
    import tomllib

    import anthropic
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8")) if secrets_path.exists() else {}
    headers = {"anthropic-workspace-id": secrets["ANTHROPIC_WORKSPACE_ID"]} if secrets.get("ANTHROPIC_WORKSPACE_ID") else None
    return anthropic.Anthropic(api_key=secrets.get("ANTHROPIC_API_KEY"), default_headers=headers), \
        secrets.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def _structured(client, model, prompt: str, schema: dict) -> dict:
    with client.messages.stream(
        model=model, max_tokens=8000,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Richiesta rifiutata dal modello")
    return json.loads(next(b.text for b in msg.content if b.type == "text"))


def generate_summaries(pages: pd.DataFrame, series: pd.DataFrame, datasets: pd.DataFrame, refresh: bool) -> dict:
    cache = json.loads(SUMMARY_CACHE.read_text(encoding="utf-8")) if SUMMARY_CACHE.exists() else {"documents": {}, "experiments": {}}
    client, model = _client()
    for fname, g in pages.groupby("file_name"):
        if fname in cache["documents"] and not refresh:
            continue
        body = "\n\n".join(f"[Pagina {r.page_number}]\n{r.text}" for r in g.itertuples())
        print(f"  sommario: {fname} ({len(body):,} caratteri)")
        cache["documents"][fname] = _structured(client, model, (
            "Sei un esperto di combustione in microgravita'. Leggi integralmente il documento NASA PSI seguente "
            f"(file: {fname}) e produci in ITALIANO: un sommario tecnico di 180-250 parole; 5-8 risultati chiave, "
            "ciascuno con i numeri di pagina che lo supportano (usa i marcatori [Pagina N]) e con i valori numerici "
            "riportati nel testo; 5-10 argomenti; le condizioni sperimentali (combustibili, diametri, pressioni, "
            "composizione dell'atmosfera, piattaforma: ISS/drop tower...). Non inventare nulla che non sia nel testo.\n\n"
            f"<documento>\n{body}\n</documento>"), DOC_SCHEMA)
        SUMMARY_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    for tag in sorted(set(pages.experiment_tag) | set(datasets.experiment_tag)):
        if tag in cache["experiments"] and not refresh:
            continue
        docs = {f: s for f, s in cache["documents"].items() if experiment_tag(f) == tag}
        ds = datasets[datasets.experiment_tag == tag].drop(columns=["raw_rows", "raw_columns"]).to_dict("records")
        print(f"  sintesi esperimento: {tag}")
        cache["experiments"][tag] = _structured(client, model, (
            f"Sintetizza in ITALIANO l'esperimento NASA PSI {tag} (150-220 parole) e 4-6 punti salienti, "
            "basandoti SOLO sui sommari dei documenti e sul catalogo dei dataset forniti. Cita i file tra parentesi.\n\n"
            f"Sommari documenti:\n{json.dumps(docs, ensure_ascii=False)}\n\nDataset:\n{json.dumps(ds, ensure_ascii=False, default=str)}"),
            EXP_SCHEMA)
        SUMMARY_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def extractive_summary(g: pd.DataFrame) -> str:
    abstract = g[g.section.str.startswith("Abstract")]
    text = " ".join((abstract if not abstract.empty else g.head(2)).text)
    return re.sub(r"\s+", " ", text)[:1200]


def write_catalogs(pages: pd.DataFrame, datasets: pd.DataFrame, cache: dict) -> None:
    docs = []
    for fname, g in pages.groupby("file_name"):
        s = cache.get("documents", {}).get(fname, {})
        docs.append({
            "experiment_tag": g.experiment_tag.iloc[0], "file_name": fname, "doc_type": g.doc_type.iloc[0],
            "title": g.title.iloc[0], "author": g.pdf_author.iloc[0] or None, "n_pages": int(len(g)),
            "outline_json": json.dumps(outline(g), ensure_ascii=False),
            "summary": s.get("summary") or extractive_summary(g),
            "summary_source": "llm" if s else "estrattivo",
            "key_findings_json": json.dumps(s.get("key_findings", []), ensure_ascii=False),
            "topics_json": json.dumps(s.get("topics", []), ensure_ascii=False),
            "experimental_conditions": s.get("experimental_conditions"),
        })
    pd.DataFrame(docs).to_parquet(PROCESSED_DIR / "documents.parquet", index=False)
    exps = []
    for tag in sorted(set(pages.experiment_tag) | set(datasets.experiment_tag)):
        s = cache.get("experiments", {}).get(tag, {})
        exps.append({"experiment_tag": tag, "overview": s.get("overview"),
                     "highlights_json": json.dumps(s.get("highlights", []), ensure_ascii=False),
                     "n_documents": int(pages[pages.experiment_tag == tag].file_name.nunique()),
                     "n_datasets": int((datasets.experiment_tag == tag).sum())})
    pd.DataFrame(exps).to_parquet(PROCESSED_DIR / "experiments.parquet", index=False)
    print(f"Catalogo documenti: {len(docs)} | esperimenti: {len(exps)}")


# ---------------------------------------------------------------------------
# 4. Metadati ChromaDB (aggiunta non distruttiva: experiment_tag, file_name, page_number, section)
# ---------------------------------------------------------------------------
def patch_chroma_metadata(pages: pd.DataFrame) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    col = client.get_collection(CHROMA_COLLECTION)
    got = col.get(include=["metadatas"])
    sec = {(r.file_name, r.page_number): (r.section, r.doc_type) for r in pages.itertuples()}
    ids, metas = [], []
    for cid, m in zip(got["ids"], got["metadatas"]):
        fname = re.split(r"[\\/]", m.get("source", ""))[-1]
        page = int(m.get("page", 0)) + 1
        section, dtype = sec.get((fname, page), ("", doc_type(fname)))
        new = {**m, "file_name": fname, "experiment_tag": experiment_tag(fname), "page_number": page,
               "section": section, "doc_type": dtype}
        if new != m:
            ids.append(cid)
            metas.append(new)
    for i in range(0, len(ids), 200):
        col.update(ids=ids[i:i + 200], metadatas=metas[i:i + 200])
    print(f"Metadati Chroma aggiornati su {len(ids)} chunk (su {len(got['ids'])})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", action="store_true", help="genera i sommari mancanti con Claude")
    ap.add_argument("--refresh-summaries", action="store_true", help="rigenera tutti i sommari")
    ap.add_argument("--skip-chroma", action="store_true", help="non aggiornare i metadati del vector DB")
    args = ap.parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    series, datasets = build_tables()
    pages = build_documents()
    cache = json.loads(SUMMARY_CACHE.read_text(encoding="utf-8")) if SUMMARY_CACHE.exists() else {}
    if args.summaries or args.refresh_summaries:
        cache = generate_summaries(pages, series, datasets, args.refresh_summaries)
    write_catalogs(pages, datasets, cache)
    if not args.skip_chroma:
        patch_chroma_metadata(pages)
    print("Fatto.")


if __name__ == "__main__":
    sys.exit(main())
