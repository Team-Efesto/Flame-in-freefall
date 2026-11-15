"""Database delle norme di sicurezza antincendio spaziale (PDF in ./data_nasa_raw/safety_pdf).

Crea un database SEPARATO da quello dei report PSI, con la stessa vettorizzazione:
  * Embeddings all-MiniLM-L6-v2 via ONNX (chromadb DefaultEmbeddingFunction), come build_vector_db.py
  * Chunk per pagina da ~1000 caratteri con overlap 200 (stessa funzione split_text)
  * Stessi metadati dei PSI (experiment_tag, file_name, page_number, doc_type, source, page, ...)
    con experiment_tag = "SAFETY" e in piu': doc_category, topics, limit_ids, requirement_ids

e in aggiunta tabelle strutturate per il confronto con i risultati delle simulazioni:

  vector_db_safety/                         ChromaDB, collezione "safety"
  data/safety/safety_pages.parquet          testo per pagina (stesso schema di doc_pages.parquet)
  data/safety/safety_documents.parquet      catalogo documenti
  data/safety/safety_limits.parquet|csv     limiti NUMERICI in unita' canoniche, con citazione verificata
  data/safety/safety_requirements.parquet|csv  requisiti NASA-STD-3001 [V2 xxxx] (testo integrale)
  data/safety/safety_numeric_mentions.parquet  tutte le grandezze numeriche con unita' trovate nei testi
                                               (candidati per estendere safety_limits)

Il confronto si fa con flame.safety.SafetyLimits (vedi docstring del modulo).

Uso:
  python build_safety_db.py             # crea (fallisce se il DB esiste gia')
  python build_safety_db.py --rebuild   # ricrea tutto
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from build_vector_db import read_document_pages, split_text, SUPPORTED_EXTS
from flame.config import (SAFETY_COLLECTION, SAFETY_DIR, SAFETY_PDF_SRC, SAFETY_TAG, SAFETY_VECTOR_DB_DIR)
from flame.safety import CANONICAL_UNITS, F_TO_K

# ---------------------------------------------------------------------------
# Limiti numerici estratti dai documenti.
# Ogni regola ha un'ancora testuale: lo script trova automaticamente file e pagina e FALLISCE se
# l'ancora non e' presente, cosi' ogni limite resta tracciabile alla fonte (niente valori "a memoria").
#
# bound_type:
#   compliant_zone -> l'intervallo [lower, upper] e' la zona conforme (fuori = NON CONFORME)
#   hazard_zone    -> l'intervallo [lower, upper] e' la zona di rischio (dentro = NON CONFORME / RISCHIO)
# lower/upper: NaN = illimitato; lower_inclusive/upper_inclusive per i bordi.
# limit_type: requirement | device_capability | experimental_threshold | design_reference
# applies_by_default=False: ambito ristretto (es. volumi inertizzati, capacita' di un dispositivo):
#   escluso dai controlli automatici, si attiva con include_all=True o limit_ids=[...]
# ---------------------------------------------------------------------------
NaN = float("nan")
LIMIT_RULES: list[dict] = [
    dict(limit_id="SAF-001", parameter="burn_length", lower=NaN, upper=15.0, bound_type="compliant_zone",
         limit_type="requirement", scope="Material acceptance: upward flammability test (NHB 8060.16)",
         description="The ignited sample must not sustain a flame that spreads beyond 15 cm.",
         file_hint="Fire Safety Applications", anchor="propagating more than a stated limit (15 cm"),
    dict(limit_id="SAF-002", parameter="air_velocity", lower=8.0, upper=20.0, bound_type="compliant_zone",
         limit_type="design_reference", scope="Nominal cabin ventilation (Shuttle, closed loop)",
         description="Nominal cabin air velocity between 8 and 20 cm/s.",
         file_hint="Fire Safety Applications", anchor="nominal air velocities between 8 and"),
    dict(limit_id="SAF-003", parameter="air_velocity", lower=8.0, upper=20.0, bound_type="hazard_zone",
         limit_type="experimental_threshold", scope="Flame spread over paper in microgravity",
         description="The highest spread hazard is at intermediate forced-flow velocities, within the Shuttle ventilation range (8-20 cm/s): there the flame can spread faster than in normal gravity.",
         file_hint="Fire Safety Applications",
         anchor="fire hazard for paper appears to be at intermediate forced-flow velocities"),
    dict(limit_id="SAF-004", parameter="oxygen_concentration", lower=15.0, upper=NaN, bound_type="hazard_zone",
         limit_type="experimental_threshold", scope="Flammability of thin materials (paper) in microgravity with intermediate forced flow",
         description="In microgravity with intermediate flow, paper can burn with O2 as low as 15%, below the flammability limit measured in normal gravity.",
         file_hint="Fire Safety Applications", anchor="may burn at oxygen concentrations as low as 15 percent"),
    dict(limit_id="SAF-005", parameter="oxygen_concentration", lower=NaN, upper=21.0, bound_type="compliant_zone",
         limit_type="design_reference", scope="Cabin reference atmosphere (sea-level air)",
         description="Baseline atmosphere = sea-level air; higher concentrations make materials more flammable. The 21% (20.95%) value is the standard air composition: the document cites 'sea-level air' without stating the number.",
         value_in_text=False,
         file_hint="Fire Safety Applications", anchor="uses standard sea-level air as its baseline atmosphere"),
    dict(limit_id="SAF-006", applies_by_default=False, parameter="oxygen_concentration", lower=NaN, upper=6.0, bound_type="compliant_zone",
         limit_type="experimental_threshold", scope="Inerted uninhabited volumes (e.g. electrical cabinets)",
         description="An inert gas that retains 6% O2 can effectively be fire-safe; the document gives no upper limit beyond 6%.",
         file_hint="Fire Safety Applications", anchor="6 percent or greater oxygen concentration may be effec"),
    dict(limit_id="SAF-007", parameter="material_separation", lower=1.3, upper=NaN, bound_type="compliant_zone",
         lower_inclusive=False, limit_type="experimental_threshold",
         scope="Spacing between flammable materials (Skylab 4, 1974)",
         description="In orbit, fire spread from one material to another across a 1.3 cm gap: distances <= 1.3 cm do not prevent spread.",
         file_hint="Fire Safety Applications", anchor="spread from one material to another over a gap of 1.3 cm"),
    dict(limit_id="SAF-008", applies_by_default=False, parameter="temperature", lower=NaN, upper=F_TO_K(2000.0), bound_type="compliant_zone",
         limit_type="device_capability", scope="Multilayer thermal protection (NASA Langley fire blanket/shelter)",
         description="Multilayer thermal blankets withstand external temperatures up to 2000 °F (1366 K).",
         file_hint="Multilayered Fire Protection", anchor="external temperatures of up to 2000 degrees Fahrenheit"),
    dict(limit_id="SAF-009", applies_by_default=False, parameter="ammonia_concentration", lower=NaN, upper=30000.0, bound_type="compliant_zone",
         limit_type="device_capability", scope="Crew emergency mask (filter cartridge)",
         description="The cartridge protects against a small fire or 30 000 ppmv (3%) ammonia for more than 15 minutes.",
         file_hint="Maturation Team", anchor="30,000 ppmv"),
    dict(limit_id="SAF-010", applies_by_default=False, parameter="exposure_time", lower=NaN, upper=15.0, bound_type="compliant_zone",
         limit_type="device_capability", scope="Crew emergency mask (guaranteed duration)",
         description="Guaranteed mask protection time: > 15 minutes (exposures up to 15 min covered).",
         file_hint="Maturation Team", anchor="ammonia for >15 min"),
]

# Parole chiave per i metadati "topics" dei chunk (filtri e ricerche per argomento)
TOPICS = {
    "ossigeno": r"\boxygen\b|\bo ?2\b",
    "propagazione_fiamma": r"flame[- ]spread|spread rate|propagat",
    "ventilazione": r"ventilat|air velocit|forced[- ]?(flow|convection)|air flow",
    "rilevazione": r"detect|smoke detector|sensor",
    "estinzione": r"extinguish|suppress|halon|water mist|foam",
    "prodotti_combustione": r"\bco\b|carbon monoxide|\bhcn\b|\bhcl\b|\bhf\b|toxic|contaminant",
    "infiammabilita_materiali": r"flammab|material",
    "microgravita": r"micro-?gravity|low[- ]gravity|partial[- ]gravity",
    "dpi_equipaggio": r"mask|breathing|protective equipment|\bppe\b",
    "incidenti": r"apollo|mir\b|accident|incident",
}

BOILERPLATE = re.compile(r"^(NASA-STD-3001 Technical Brief|OCHMO-TB-|NASA Office of the Chief|This Technical Brief|"
                         r"It does not supersede|\d+/\d+/\d{4}|Rev [A-Z]$|\d+$|\.$)", re.IGNORECASE)
NUM_UNIT_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(%|percent|ppmv|ppm|cm/s|mm/s|m/s|kPa|psia|psig|psi|mmHg|atm|"
    r"degrees Fahrenheit|°F|°C|K\b|cm\b|mm\b|m\b|min\b|minutes|seconds|s\b|hours|kg\b|kW\b|W\b)", re.IGNORECASE)
REQ_ID_RE = re.compile(r"\[(V[12]) (\d{4,5})\]")


LIGATURES = {"\u00ad": "", "\u2010": "-", "\u2011": "-", "\u00a0": " ", "\ufb00": "ff", "\ufb01": "fi",
             "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def normalize_text(text: str) -> str:
    """Pulisce il testo estratto: legature, trattini morbidi, spazi multipli e PDF 'una parola per riga'."""
    for a, b in LIGATURES.items():
        text = text.replace(a, b)
    text = re.sub(r"(?<=\w)-{2,}(?=\w)", "-", text)       # "MCC--Dulles" (trattino morbido + trattino)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    nonempty = [ln for ln in lines if ln]
    single = sum(1 for ln in nonempty if " " not in ln)
    if len(nonempty) > 20 and single / len(nonempty) > 0.8:  # una parola per riga (es. articolo su Mir)
        return re.sub(r"\s+([,.;:])", r"\1", " ".join(nonempty))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def doc_category(fname: str) -> str:
    low = fname.lower()
    if "ochmo" in low or "technical brief" in low:
        return "Norma tecnica (NASA-STD-3001)"
    if "maturation team" in low:
        return "Presentazione"
    if "fire safety applications" in low:
        return "Report tecnico"
    if "years ago" in low:
        return "Rapporto incidente"
    return "Scheda tecnologia (spinoff)"


def doc_title(fname: str) -> str:
    return re.sub(r"^NASA_", "", Path(fname).stem).replace("_s ", "'s ").replace("_", " ").strip()


def page_section(text: str, category: str) -> str:
    for ln in text.splitlines()[:15]:
        ln = ln.strip()
        if 3 <= len(ln) <= 80 and not BOILERPLATE.match(ln) and re.search(r"[A-Za-z]{3}", ln):
            return ln
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return first[:80].rsplit(" ", 1)[0] + "…" if len(first) > 80 else first


def topics_of(text: str) -> list[str]:
    low = text.lower()
    return [k for k, rx in TOPICS.items() if re.search(rx, low)]


def read_pages(src: Path) -> pd.DataFrame:
    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        raise SystemExit(f"Nessun documento (PDF/TXT/DOCX) trovato in {src}")
    rows = []
    for f in files:
        pages = [normalize_text(t) for t in read_document_pages(f)]
        cat = doc_category(f.name)
        print(f"  {f.name}: {len(pages)} pagine")
        for i, text in enumerate(pages):
            rows.append({"experiment_tag": SAFETY_TAG, "file_name": f.name, "doc_type": cat, "title": doc_title(f.name),
                         "page_number": i + 1, "page_label": str(i + 1), "section": page_section(text, cat),
                         "n_chars": len(text), "text": text, "pdf_author": "", "total_pages": len(pages)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Limiti, requisiti, menzioni numeriche
# ---------------------------------------------------------------------------
def sentence_around(text: str, anchor: str) -> str:
    """Frase (o finestra) del testo originale che contiene l'ancora, per la citazione."""
    t = re.sub(r"\s+", " ", text)
    i = t.lower().find(flat(anchor))
    start = max([t.rfind(sep, 0, i) + len(sep) for sep in (". ", "• ", " – ", "♦ ")] + [i - 250, 0])
    end = t.find(". ", i + len(anchor))
    end = len(t) if end < 0 or end - i > 400 else end + 1
    return t[start:end].strip()


def build_limits(pages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in LIMIT_RULES:
        cand = pages[pages.file_name.str.contains(rule["file_hint"], case=False, regex=False)]
        hit = next((r for r in cand.itertuples() if flat(rule["anchor"]) in flat(r.text)), None)
        if hit is None:
            raise SystemExit(f"{rule['limit_id']}: ancora non trovata nei PDF ({rule['anchor']!r}) - limite non verificabile")
        unit = CANONICAL_UNITS[rule["parameter"]]
        rows.append({
            "limit_id": rule["limit_id"], "parameter": rule["parameter"], "unit": unit,
            "lower": rule["lower"], "upper": rule["upper"],
            "lower_inclusive": rule.get("lower_inclusive", True), "upper_inclusive": rule.get("upper_inclusive", True),
            "bound_type": rule["bound_type"], "limit_type": rule["limit_type"], "scope": rule["scope"],
            "applies_by_default": rule.get("applies_by_default", True),
            "description": rule["description"], "value_in_text": rule.get("value_in_text", True),
            "file_name": hit.file_name, "page_number": int(hit.page_number), "doc_type": hit.doc_type,
            "quote": sentence_around(hit.text, rule["anchor"]),
            "citation": f"[Source: {SAFETY_TAG} | {hit.file_name}, p. {int(hit.page_number)}]",
        })
    return pd.DataFrame(rows)


def build_requirements(pages: pd.DataFrame) -> pd.DataFrame:
    """Requisiti NASA-STD-3001 '[V2 xxxx] Titolo ... shall ...' (testo integrale nelle pagine di back-up)."""
    rows = []
    for fname, g in pages.groupby("file_name"):
        g = g.sort_values("page_number")
        # titoli dall'elenco in copertina: "[V2 6024] Combustion Monitoring and Alerting"
        titles = {}
        for m in re.finditer(r"\[(V[12]) (\d{4,5})\]\s*([^\[]+?)(?=\s*\[V[12] |\s*Executive Summary|$)",
                             re.sub(r"\s+", " ", g.text.iloc[0])):
            titles[f"{m.group(1)} {m.group(2)}"] = re.split(r"\s+NASA-STD-3001", m.group(3))[0].strip()
        for r in g.itertuples():
            t = re.sub(r"\s+", " ", r.text)
            parts = REQ_ID_RE.split(t)  # [pre, vol, num, body, vol, num, body, ...]
            for vol, num, body in zip(parts[1::3], parts[2::3], parts[3::3]):
                if " shall " not in body:
                    continue
                body = re.split(r"\s(?:Referenced Technical Requirements|All referenced tables|View the current|"
                                r"NASA-STD-3001 Volume \d)", body)[0].strip()
                body = re.sub(r"\s\d{1,2}\s\d{1,2}/\d{1,2}/\d{4}.*$", "", body).strip()
                rid = f"{vol} {num}"
                title = titles.get(rid, "")
                text = body[len(title):].strip() if title and body.lower().startswith(flat(title)[:20]) else body
                rows.append({"requirement_id": rid, "standard": "NASA-STD-3001 Volume " + vol[1], "title": title,
                             "text": text, "related_limit_ids": "", "file_name": fname, "page_number": int(r.page_number),
                             "citation": f"[Source: {SAFETY_TAG} | {fname}, p. {int(r.page_number)}]"})
    df = pd.DataFrame(rows).drop_duplicates("requirement_id") if rows else pd.DataFrame()
    return df


def build_numeric_mentions(pages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in pages.itertuples():
        t = re.sub(r"\s+", " ", r.text)
        for m in NUM_UNIT_RE.finditer(t):
            rows.append({"file_name": r.file_name, "page_number": int(r.page_number),
                         "value": float(m.group(1).replace(",", "")), "unit_raw": m.group(2),
                         "context": t[max(0, m.start() - 120):m.end() + 120].strip()})
    return pd.DataFrame(rows)


def attach_ids(text: str, limits: pd.DataFrame, file_name: str, page: int) -> tuple[str, str]:
    lim = limits[(limits.file_name == file_name) & (limits.page_number == page)].limit_id.tolist()
    reqs = sorted({f"{v} {n}" for v, n in REQ_ID_RE.findall(text)})
    return ",".join(lim), ",".join(reqs)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(SAFETY_PDF_SRC))
    ap.add_argument("--db-dir", default=str(SAFETY_VECTOR_DB_DIR))
    ap.add_argument("--out-dir", default=str(SAFETY_DIR))
    ap.add_argument("--rebuild", action="store_true", help="elimina e ricrea DB e tabelle esistenti")
    args = ap.parse_args()
    db_dir, out_dir = Path(args.db_dir), Path(args.out_dir)

    if db_dir.exists() and any(db_dir.iterdir()):
        if not args.rebuild:
            print(f"Il database {db_dir} esiste gia'. Usa --rebuild per ricrearlo.")
            return 1
        shutil.rmtree(db_dir)

    print(f"Lettura documenti da {args.pdf_dir}")
    pages = read_pages(Path(args.pdf_dir))
    limits = build_limits(pages)
    reqs = build_requirements(pages)
    mentions = build_numeric_mentions(pages)

    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    client = chromadb.PersistentClient(path=str(db_dir))
    col = client.create_collection(SAFETY_COLLECTION, embedding_function=DefaultEmbeddingFunction(),
                                   metadata={"description": "Norme e documenti NASA di sicurezza antincendio spaziale"})
    ids, docs, metas = [], [], []
    for r in pages.itertuples():
        for ci, chunk in enumerate(split_text(r.text)):
            lim_ids, req_ids = attach_ids(chunk, limits, r.file_name, r.page_number)
            ids.append(f"{re.sub(r'[^A-Za-z0-9]+', '_', Path(r.file_name).stem)}_p{r.page_number}_c{ci}")
            docs.append(chunk)
            metas.append({"source": str(Path("data_nasa_raw") / "safety_pdf" / r.file_name), "page": r.page_number - 1,
                          "page_label": r.page_label, "total_pages": r.total_pages, "file_name": r.file_name,
                          "experiment_tag": SAFETY_TAG, "page_number": r.page_number, "doc_type": r.doc_type,
                          "title": r.title, "author": "", "category": "safety", "doc_category": r.doc_type,
                          "section": r.section, "topics": ",".join(topics_of(chunk)),
                          "limit_ids": lim_ids, "requirement_ids": req_ids})
    for i in range(0, len(ids), 128):
        col.add(ids=ids[i:i + 128], documents=docs[i:i + 128], metadatas=metas[i:i + 128])

    out_dir.mkdir(parents=True, exist_ok=True)
    pages_out = pages.drop(columns=["total_pages"])
    pages_out.to_parquet(out_dir / "safety_pages.parquet", index=False)
    n_chunks = pd.Series([m["file_name"] for m in metas]).value_counts()
    docs_df = (pages.groupby("file_name").agg(title=("title", "first"), doc_type=("doc_type", "first"),
                                              n_pages=("page_number", "max"), n_chars=("n_chars", "sum"))
               .reset_index())
    docs_df["experiment_tag"] = SAFETY_TAG
    docs_df["n_chunks"] = docs_df.file_name.map(n_chunks).fillna(0).astype(int)
    docs_df["n_limits"] = docs_df.file_name.map(limits.file_name.value_counts()).fillna(0).astype(int)
    docs_df.to_parquet(out_dir / "safety_documents.parquet", index=False)
    limits.to_parquet(out_dir / "safety_limits.parquet", index=False)
    limits.to_csv(out_dir / "safety_limits.csv", index=False, encoding="utf-8")
    if not reqs.empty:
        reqs.to_parquet(out_dir / "safety_requirements.parquet", index=False)
        reqs.to_csv(out_dir / "safety_requirements.csv", index=False, encoding="utf-8")
    mentions.to_parquet(out_dir / "safety_numeric_mentions.parquet", index=False)

    print(f"Indicizzati {len(ids)} chunk da {pages.file_name.nunique()} documenti in {db_dir} (collezione '{SAFETY_COLLECTION}')")
    print(f"Limiti numerici: {len(limits)} | requisiti NASA-STD-3001: {len(reqs)} | menzioni numeriche: {len(mentions)}")
    print(f"Tabelle in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
