"""Accesso ai dati processati: caricamento Parquet e motore SQL read-only (DuckDB in memoria).

Le query SQL generate dal modello vengono eseguite su una connessione DuckDB isolata:
  * sola lettura di DataFrame registrati in memoria (nessun file, nessuna rete);
  * enable_external_access=false + lock_configuration=true (impossibile riattivarlo);
  * una sola istruzione SELECT/WITH, limite di righe.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property

import duckdb
import pandas as pd

from .config import PROCESSED_DIR, RAW_TABLES_DIR

MAX_ROWS = 50_000
_FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|export|import|install|load|pragma|set|reset|create|insert|update|delete|drop|alter|"
    r"call|checkpoint|vacuum|read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|glob|httpfs)\b",
    re.IGNORECASE,
)

TABLE_DOCS = {
    "points": "Una riga per punto misurato/simulato. Colonne: series_id, experiment_tag, dataset, source_file, "
              "series_label, kind (experiment|model|simulation|unspecified), kind_inferred, fuel, pressure_atm, d0_mm, "
              "x_var, x_unit, x_column, y_var, y_unit, y_column, y_description, point_idx, x, y.",
    "series": "Catalogo serie (una riga per serie): stessi metadati di points + x_description, n_points, x_min, x_max, y_min, y_max.",
    "datasets": "Una riga per file sorgente/figura: dataset, experiment_tag, source_file, role, journal, year, figure, fuel, "
                "d0_mm, pressure_atm, x_o2, x_n2 (testo, come nel ReadMe), reference_doi, n_series, n_points.",
    "documents": "Catalogo report PDF: experiment_tag, file_name, doc_type, title, author, n_pages, summary.",
}


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool


class DataStore:
    def __init__(self, processed_dir=PROCESSED_DIR):
        self.dir = processed_dir
        self.points = pd.read_parquet(self.dir / "points.parquet")
        self.series = pd.read_parquet(self.dir / "series.parquet")
        self.datasets = pd.read_parquet(self.dir / "datasets.parquet")
        self.documents = pd.read_parquet(self.dir / "documents.parquet")
        self.pages = pd.read_parquet(self.dir / "doc_pages.parquet")
        exp_path = self.dir / "experiments.parquet"
        self.experiments = pd.read_parquet(exp_path) if exp_path.exists() else pd.DataFrame()

    # ------------------------------------------------------------------ SQL
    def _connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        con.register("points", self.points)
        con.register("series", self.series)
        con.register("datasets", self.datasets)
        con.register("documents", self.documents.drop(columns=["outline_json", "key_findings_json", "topics_json"], errors="ignore"))
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        return con

    @staticmethod
    def validate_sql(sql: str) -> str:
        s = sql.strip().rstrip(";").strip()
        if not s:
            raise ValueError("Query vuota.")
        if ";" in s:
            raise ValueError("E' consentita una sola istruzione SQL.")
        if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
            raise ValueError("Sono consentite solo query SELECT (o WITH ... SELECT).")
        stripped = re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", s)  # ignora stringhe/identificatori
        m = _FORBIDDEN.search(stripped)
        if m:
            raise ValueError(f"Parola chiave non consentita: {m.group(0)}")
        return s

    def query(self, sql: str, max_rows: int = MAX_ROWS) -> QueryResult:
        s = self.validate_sql(sql)
        con = self._connection()
        try:
            df = con.execute(f"SELECT * FROM ({s}) AS _q LIMIT {max_rows + 1}").df()
        finally:
            con.close()
        truncated = len(df) > max_rows
        return QueryResult(df.head(max_rows), truncated)

    # ------------------------------------------------------------ helpers
    def raw_table(self, dataset: str) -> pd.DataFrame:
        path = RAW_TABLES_DIR / f"{dataset}.parquet"
        if not path.exists():
            raise KeyError(f"Dataset sconosciuto: {dataset}")
        df = pd.read_parquet(path)
        for c in df.columns:
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().sum() == df[c].notna().sum():
                df[c] = num
        return df

    @property
    def experiment_tags(self) -> list[str]:
        tags = set(self.datasets.experiment_tag) | set(self.documents.experiment_tag)
        return sorted(t for t in tags if t != "UNKNOWN")

    @cached_property
    def catalog_text(self) -> str:
        """Catalogo compatto (stabile, quindi cacheabile) incluso nel prompt di sistema."""
        lines = ["## Tabelle SQL disponibili"]
        lines += [f"- `{k}`: {v}" for k, v in TABLE_DOCS.items()]
        lines.append("\n## Dataset (file sorgente → figura dell'articolo)")
        for r in self.datasets.itertuples():
            if r.role == "readme":
                lines.append(f"- {r.dataset} [{r.experiment_tag}] file `{r.source_file}`: indice delle figure (ReadMe).")
                continue
            lines.append(
                f"- {r.dataset} [{r.experiment_tag}] file `{r.source_file}` | {r.journal or ''} {r.year or ''} {r.figure or ''} | "
                f"fuel={r.fuel}; d0_mm={r.d0_mm}; P_atm={r.pressure_atm}; XO2={r.x_o2}; DOI {r.reference_doi} | {r.n_series} serie")
        lines.append("\n## Serie (series_id | etichetta | kind | fuel, P atm, d0 mm | x_var→y_var [unita'] | colonna y | n | range y)")
        for r in self.series.itertuples():
            cond = ", ".join(str(v) for v in (r.fuel, r.pressure_atm, r.d0_mm) if v is not None and v == v)
            lines.append(
                f"- {r.series_id} | {r.series_label} | {r.kind}{'*' if r.kind_inferred else ''} | {cond} | "
                f"{r.x_var}[{r.x_unit}]→{r.y_var}[{r.y_unit}] | `{r.y_column}` | {r.n_points} | {r.y_min:.4g}…{r.y_max:.4g}")
        lines.append("(* = tipo dedotto: serie non etichettata in un file che etichetta esplicitamente modelli/simulazioni)")
        lines.append("\n## Report PDF")
        for r in self.documents.itertuples():
            lines.append(f"- [{r.experiment_tag}] `{r.file_name}` — {r.doc_type}, {r.n_pages} pagine — {r.title}")
        return "\n".join(lines)

    def document_overview(self, experiment_tag: str | None = None, file_name: str | None = None) -> str:
        docs = self.documents
        if experiment_tag:
            docs = docs[docs.experiment_tag == experiment_tag]
        if file_name:
            docs = docs[docs.file_name == file_name]
        out = []
        if not self.experiments.empty:
            for e in self.experiments.itertuples():
                if (not experiment_tag or e.experiment_tag == experiment_tag) and e.overview:
                    out.append(f"# Esperimento {e.experiment_tag}\n{e.overview}\nPunti salienti: "
                               + "; ".join(json.loads(e.highlights_json)))
        for r in docs.itertuples():
            findings = json.loads(r.key_findings_json or "[]")
            outline = json.loads(r.outline_json or "[]")
            out.append("\n".join([
                f"## {r.file_name} [{r.experiment_tag}] — {r.doc_type}, {r.n_pages} pagine",
                f"Titolo: {r.title}" + (f" | Autore: {r.author}" if r.author else ""),
                f"Sommario ({r.summary_source}): {r.summary}",
                *(f"- {f['finding']} (pp. {', '.join(map(str, f['pages']))})" for f in findings),
                f"Condizioni sperimentali: {r.experimental_conditions}" if r.experimental_conditions else "",
                "Indice: " + "; ".join(f"p.{o['start_page']} {o['section']}" for o in outline[:60]),
            ]))
        return "\n\n".join(out) or "Nessun documento trovato con questi filtri."
