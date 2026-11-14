"""Agente Claude con tool use: interroga l'intero dataset (SQL su DuckDB), i report (retrieval
ibrido + lettura di pagine intere + sommari) e produce tabelle e grafici renderizzati dalla UI.

I risultati completi (DataFrame, figure) restano lato applicazione come "artefatti"; al modello
arrivano solo anteprime e statistiche, cosi' i token restano contenuti anche su risultati grandi.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import anthropic
import pandas as pd

from .charts import CHART_TYPES, build_chart
from .data_store import DataStore
from .export import sources_of
from .retrieval import ReportIndex

MAX_TOOL_ROUNDS = 10
PREVIEW_ROWS = 40
MAX_TOOL_CHARS = 9000

TOOLS = [
    {
        "name": "query_data",
        "description": (
            "Esegue una query SQL (DuckDB, sola lettura, una sola SELECT/WITH) sull'INTERO dataset numerico: tabelle "
            "`points`, `series`, `datasets`, `documents`. Usalo per ogni domanda quantitativa (valori, confronti, "
            "filtri, aggregazioni, statistiche, regressioni con regr_slope ecc.). Restituisce un result_id, il numero "
            "di righe, le colonne e un'anteprima. Il risultato completo e' disponibile per create_chart e, se "
            "display=true, viene mostrato all'utente come tabella scaricabile (CSV/XLSX). Includi sempre le colonne "
            "series_id e source_file quando estrai punti, per poter citare la fonte."),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Query SELECT. Le stringhe vanno tra apici singoli."},
                "title": {"type": "string", "description": "Titolo breve e descrittivo della tabella risultante."},
                "display": {"type": "boolean", "description": "true se la tabella e' un risultato da mostrare all'utente; false per query esplorative/di supporto."},
            },
            "required": ["sql", "title", "display"],
        },
    },
    {
        "name": "create_chart",
        "description": (
            "Crea un grafico interattivo (Plotly) a partire da un risultato di query_data e lo mostra all'utente. "
            "Usalo quando la domanda riguarda andamenti, confronti tra serie/esperimenti, relazioni o distribuzioni. "
            "Un solo asse y: se le grandezze hanno unita' diverse usa `facet` (pannelli separati) o grafici distinti. "
            "Usa `color` per distinguere le serie (max 8 categorie)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "chart_type": {"type": "string", "enum": list(CHART_TYPES)},
                "x": {"type": "string", "description": "Colonna asse x"},
                "y": {"type": "string", "description": "Colonna asse y (non serve per histogram)"},
                "color": {"type": "string", "description": "Colonna categorica per il colore/legenda"},
                "facet": {"type": "string", "description": "Colonna per pannelli separati"},
                "error_y": {"type": "string", "description": "Colonna con l'ampiezza della barra d'errore"},
                "title": {"type": "string"},
                "x_label": {"type": "string", "description": "Etichetta asse x con unita', es. 'Pressione [atm]'"},
                "y_label": {"type": "string", "description": "Etichetta asse y con unita'"},
                "log_x": {"type": "boolean"},
                "log_y": {"type": "boolean"},
            },
            "required": ["result_id", "chart_type", "x", "title"],
        },
    },
    {
        "name": "search_reports",
        "description": (
            "Ricerca ibrida (semantica + parole chiave) nei report PDF. Restituisce le pagine piu' pertinenti con "
            "file, numero di pagina e sezione, per citare con precisione. Usa query in inglese (i report sono in "
            "inglese) e specifiche; ripeti con formulazioni diverse se serve."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "experiment_tag": {"type": "string", "description": "Filtra per codice, es. PSI-117"},
                "file_name": {"type": "string", "description": "Filtra su un singolo PDF"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_report_pages",
        "description": "Legge il testo integrale di un intervallo di pagine (max 8) di un report: usalo per sezioni/capitoli interi (vedi l'indice in get_document_overview) quando servono contesto completo, tabelle o derivazioni.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string"},
                "start_page": {"type": "integer", "minimum": 1},
                "end_page": {"type": "integer", "minimum": 1},
            },
            "required": ["file_name", "start_page", "end_page"],
        },
    },
    {
        "name": "get_document_overview",
        "description": "Sommari strutturati, risultati chiave con pagine, condizioni sperimentali e indice delle sezioni dei report (e sintesi per esperimento PSI-XXX). Primo passo per domande di sintesi, panoramica o confronto tra documenti.",
        "input_schema": {
            "type": "object",
            "properties": {"experiment_tag": {"type": "string"}, "file_name": {"type": "string"}},
        },
    },
    {
        "name": "get_raw_table",
        "description": "Restituisce la tabella ORIGINALE (formato largo, come nel file CSV/XLSX NASA) di un dataset, utile se l'utente chiede il file/tabella sorgente.",
        "input_schema": {
            "type": "object",
            "properties": {"dataset": {"type": "string"}, "display": {"type": "boolean"}},
            "required": ["dataset", "display"],
        },
    },
]

SYSTEM_TEMPLATE = """Sei l'assistente scientifico di "Flame in Freefall", piattaforma di analisi dei dati NASA Physical Science Informatics (PSI) sulla combustione in microgravita' (gocce isolate, cool flames, estinzione).

# Come lavori
- Hai accesso COMPLETO ai dati tramite gli strumenti: non basarti su memoria o supposizioni.
- Domande quantitative (valori, confronti, filtri, "estrai una tabella", statistiche): usa `query_data` sul dataset completo; non stimare numeri dal testo dei report se esistono nei dati.
- Andamenti, confronti tra serie o condizioni, distribuzioni: dopo `query_data` chiama `create_chart` con un grafico pertinente (etichette assi con unita').
- Richieste di tabelle/estrazioni: `query_data` con display=true (l'utente potra' scaricarla in CSV/XLSX). Usa display=false per query esplorative.
- Domande puramente teoriche o concettuali: rispondi dai report (`search_reports`, `read_report_pages`, `get_document_overview`) SENZA grafici o tabelle.
- Domande di sintesi/panoramica: parti da `get_document_overview`: i risultati chiave li' riportati sono gia' estratti dal testo con le pagine e bastano per una sintesi. Usa `search_reports`/`read_report_pages` solo per verificare o approfondire punti specifici (al massimo 2-3 letture mirate), per contenere i token.
- Se un dato richiesto non esiste (es. altri esperimenti PSI, velocita' di propagazione della fiamma al variare dell'O2), dichiaralo chiaramente e proponi l'analisi piu' vicina possibile con i dati disponibili. Non inventare mai valori.
- I dati tabellari sono "figure data": valori digitalizzati/esportati dalle figure degli articoli citati (DOI nel catalogo); le serie di tipo model/simulation sono risultati numerici, non misure.

# Citazioni (OBBLIGATORIE)
- Ogni affermazione tratta dai report: [Fonte: PSI-XXX | nome_file.pdf, p. N]
- Ogni valore tratto dai dati: [Dati: PSI-XXX | nome_file.csv, serie <series_id>] (aggiungi figura/DOI quando utile).
- Scrivi SEMPRE il nome file completo ed esatto (non abbreviarlo, non "la tesi"), e il numero di pagina del PDF restituito dagli strumenti.
- Usa solo file, pagine e serie effettivamente restituiti dagli strumenti.

# Stile
- Rispondi nella lingua dell'utente (predefinito: italiano), in Markdown conciso e tecnico.
- Non ricopiare nel testo tabelle lunghe: l'interfaccia mostra gia' tabelle e grafici; commentali e riassumi i numeri chiave.
- Unita' di misura sempre esplicite.

# Suggerimenti SQL
- Filtra per grandezza con `y_var`/`x_var` (codici canonici) e per serie con `series_id`; `kind` distingue experiment/model/simulation.
- Esempio: SELECT series_id, series_label, kind, source_file, x AS pressure_atm, y AS dext_mm FROM points WHERE y_var = 'extinction_diameter_cool_flame' ORDER BY series_id, x
- Per i grafici includi una colonna leggibile per `color` (es. series_label || ' (' || kind || ')').

# Catalogo dati
{catalog}
"""


@dataclass
class ResultTable:
    id: str
    title: str
    df: pd.DataFrame
    sql: str | None
    display: bool
    truncated: bool = False

    @property
    def sources(self) -> list[str]:
        return sources_of(self.df)


@dataclass
class ChartArtifact:
    title: str
    result_id: str
    figure: object  # plotly Figure


@dataclass
class TurnArtifacts:
    tables: dict[str, ResultTable] = field(default_factory=dict)
    charts: list[ChartArtifact] = field(default_factory=list)
    doc_hits: dict[tuple, dict] = field(default_factory=dict)  # (file, page) -> info
    trace: list[dict] = field(default_factory=list)

    @property
    def displayed_tables(self) -> list[ResultTable]:
        return [t for t in self.tables.values() if t.display]

    @property
    def data_sources(self) -> list[str]:
        out = set()
        for t in self.tables.values():
            out.update(t.sources)
        return sorted(out)


@dataclass
class AgentResult:
    text: str
    artifacts: TurnArtifacts
    stop_reason: str | None
    usage: dict


def _preview(df: pd.DataFrame) -> str:
    with pd.option_context("display.max_columns", 30, "display.width", 250, "display.max_colwidth", 60):
        head = df.head(PREVIEW_ROWS).to_string(index=False)
        txt = f"{len(df)} righe x {df.shape[1]} colonne. Colonne: {list(df.columns)}\nAnteprima:\n{head}"
        num = df.select_dtypes("number")
        if len(df) > PREVIEW_ROWS and not num.empty:
            txt += "\nStatistiche colonne numeriche:\n" + num.describe().T[["count", "mean", "min", "max"]].to_string()
    return txt[:MAX_TOOL_CHARS]


class FlameAgent:
    def __init__(self, client: anthropic.Anthropic, store: DataStore, index: ReportIndex, model: str,
                 effort: str | None = None):
        self.client, self.store, self.index, self.model, self.effort = client, store, index, model, effort
        self.system = [{"type": "text", "text": SYSTEM_TEMPLATE.format(catalog=store.catalog_text),
                        "cache_control": {"type": "ephemeral"}}]

    # ----------------------------------------------------------------- tools
    def _run_tool(self, name: str, args: dict, art: TurnArtifacts) -> str:
        if name == "query_data":
            res = self.store.query(args["sql"])
            rid = f"R{len(art.tables) + 1}"
            art.tables[rid] = ResultTable(rid, args.get("title") or rid, res.df, args["sql"], bool(args.get("display", True)), res.truncated)
            note = " (TRONCATO al limite di righe)" if res.truncated else ""
            return f"result_id={rid}{note}. " + _preview(res.df)
        if name == "create_chart":
            t = art.tables.get(args["result_id"])
            if t is None:
                raise ValueError(f"result_id sconosciuto: {args['result_id']}. Disponibili: {list(art.tables)}")
            fig = build_chart(t.df, args["chart_type"], args["x"], args.get("y"), color=args.get("color"),
                              facet=args.get("facet"), title=args.get("title"), x_label=args.get("x_label"),
                              y_label=args.get("y_label"), log_x=bool(args.get("log_x")), log_y=bool(args.get("log_y")),
                              error_y=args.get("error_y"))
            art.charts.append(ChartArtifact(args.get("title") or t.title, t.id, fig))
            return f"Grafico creato e mostrato all'utente ({args['chart_type']}, {len(t.df)} punti)."
        if name == "search_reports":
            fname = args.get("file_name")
            if fname:
                fname = self.index.resolve_file(fname) or fname
            hits = self.index.search(args["query"], experiment_tag=args.get("experiment_tag") or None,
                                     file_name=fname, top_k=int(args.get("top_k") or 6))
            if not hits:
                return "Nessun risultato. Prova un'altra formulazione (in inglese) o rimuovi i filtri."
            blocks = []
            for h in hits:
                art.doc_hits.setdefault((h.file_name, h.page_number), {"tag": h.experiment_tag, "section": h.section, "text": h.text})
                blocks.append(f"{h.citation} | sezione: {h.section or 'n/d'} | match: {h.matched_by}\n{h.text}")
            return "\n\n---\n\n".join(blocks)[:MAX_TOOL_CHARS * 2]
        if name == "read_report_pages":
            fname = self.index.resolve_file(args["file_name"])
            if not fname:
                raise ValueError(f"File non trovato: {args['file_name']}. File: {self.index.pages.file_name.unique().tolist()}")
            g = self.index.read_pages(fname, int(args["start_page"]), int(args["end_page"]))
            if g.empty:
                raise ValueError("Nessuna pagina in questo intervallo.")
            out = []
            for r in g.itertuples():
                art.doc_hits.setdefault((r.file_name, int(r.page_number)), {"tag": r.experiment_tag, "section": r.section, "text": r.text})
                out.append(f"[Fonte: {r.experiment_tag} | {r.file_name}, p. {r.page_number}] sezione: {r.section or 'n/d'}\n{r.text}")
            return "\n\n---\n\n".join(out)[:24000]
        if name == "get_document_overview":
            fname = args.get("file_name")
            if fname:
                fname = self.index.resolve_file(fname) or fname
            return self.store.document_overview(args.get("experiment_tag") or None, fname)[:20000]
        if name == "get_raw_table":
            ds = args["dataset"]
            df = self.store.raw_table(ds)
            src = self.store.datasets.loc[self.store.datasets.dataset == ds, "source_file"]
            df.insert(0, "source_file", src.iloc[0] if len(src) else ds)
            rid = f"R{len(art.tables) + 1}"
            art.tables[rid] = ResultTable(rid, f"Tabella originale — {ds}", df, None, bool(args.get("display", True)))
            return f"result_id={rid}. " + _preview(df)
        raise ValueError(f"Strumento sconosciuto: {name}")

    @staticmethod
    def _describe(name: str, args: dict) -> str:
        return {
            "query_data": lambda: f"🧮 Query sui dati: {args.get('title', '')}",
            "create_chart": lambda: f"📈 Grafico: {args.get('title', '')}",
            "search_reports": lambda: f"🔎 Ricerca nei report: “{args.get('query', '')}”",
            "read_report_pages": lambda: f"📖 Lettura {args.get('file_name', '')[:50]} pp. {args.get('start_page')}–{args.get('end_page')}",
            "get_document_overview": lambda: "🗂️ Sommari e indici dei report",
            "get_raw_table": lambda: f"📄 Tabella originale {args.get('dataset', '')}",
        }.get(name, lambda: name)()

    # ------------------------------------------------------------------ loop
    def run(self, history: list[dict], user_text: str, experiment_filter: str | None = None,
            on_event: Callable[[str], None] | None = None) -> AgentResult:
        art = TurnArtifacts()
        note = (f"\n\n[Filtro attivo nell'interfaccia: {experiment_filter}. Limita la ricerca a questo esperimento salvo "
                f"richiesta esplicita di confronto.]") if experiment_filter else ""
        messages = list(history) + [{"role": "user", "content": user_text + note}]
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        response = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.messages.create(
                model=self.model, max_tokens=16000, system=self.system, tools=TOOLS, messages=messages, **extra)
            for k in usage:
                usage[k] += getattr(response.usage, k, 0) or 0
            if response.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = block.input if isinstance(block.input, dict) else json.loads(block.input or "{}")
                desc = self._describe(block.name, args)
                if on_event:
                    on_event(desc)
                try:
                    out = self._run_tool(block.name, args, art)
                    art.trace.append({"tool": block.name, "label": desc, "input": args, "ok": True})
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
                except Exception as e:  # l'errore torna al modello, che puo' correggersi (es. SQL)
                    art.trace.append({"tool": block.name, "label": desc, "input": args, "ok": False, "error": str(e)})
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"Errore: {e}", "is_error": True})
            messages.append({"role": "user", "content": results})
        text = "\n".join(b.text for b in (response.content if response else []) if b.type == "text").strip()
        stop = response.stop_reason if response else None
        if stop == "refusal":
            text = "⚠️ La richiesta non puo' essere elaborata dal modello. Prova a riformularla."
        elif stop == "tool_use":
            text = (text + "\n\n" if text else "") + "⚠️ Analisi interrotta: raggiunto il numero massimo di passaggi. Prova a restringere la domanda."
        elif stop == "max_tokens":
            text += "\n\n⚠️ Risposta troncata per lunghezza."
        return AgentResult(text or "_(nessuna risposta testuale)_", art, stop, usage)

    @staticmethod
    def history_entry(result: AgentResult) -> str:
        """Testo dell'assistente salvato nella cronologia, con un promemoria compatto degli artefatti
        (permette domande di follow-up senza reinviare i dati)."""
        notes = [f"{t.id} '{t.title}' ({len(t.df)} righe) SQL: {t.sql}" for t in result.artifacts.tables.values() if t.sql]
        notes += [f"grafico '{c.title}' da {c.result_id}" for c in result.artifacts.charts]
        return result.text + ("\n\n[Artefatti del turno precedente: " + " | ".join(notes) + "]" if notes else "")
