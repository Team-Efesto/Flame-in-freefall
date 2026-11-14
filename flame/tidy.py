"""Conversione delle tabelle NASA PSI "figure data" (formato largo, piu' serie affiancate)
in un formato lungo e interrogabile: una riga per punto (x, y) di ogni serie.

Le tabelle sorgente contengono, per ogni figura di un articolo, piu' coppie di colonne
(x, y) affiancate, separate da colonne vuote e precedute da colonne-etichetta
("EXP", "Model", "Simulation", "0.5 atm", "do 4 mm", ...). L'algoritmo e' generico
(nessuna regola specifica per file) cosi' funziona anche su nuovi file PSI-XXX.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .config import experiment_tag

# ---------------------------------------------------------------------------
# Grandezze canoniche: permettono query trasversali tra dataset diversi
# (es. "tutte le serie di radianza") anche se le intestazioni originali variano.
# (regex sul nome base minuscolo, codice canonico, unita', descrizione)
# ---------------------------------------------------------------------------
QUANTITY_RULES: list[tuple[str, str, str, str]] = [
    (r"^t/\(?do\)?\^?2|^t/do\^?2", "time_normalized", "s/mm^2", "Tempo normalizzato t/d0^2"),
    (r"^(t|time)(\s*\((s|sec)\)|\(sec\))?$", "time", "s", "Tempo"),
    (r"^(p|pressure)(\s*\(atm\))?$", "pressure", "atm", "Pressione ambiente"),
    (r"^do(\s*\(mm\))?$", "initial_droplet_diameter", "mm", "Diametro iniziale della goccia d0"),
    (r"^d\^2", "droplet_diameter_squared", "mm^2", "Quadrato del diametro della goccia d^2"),
    (r"^\(d/do\)\^?2", "droplet_diameter_squared_normalized", "-", "Diametro goccia normalizzato (d/d0)^2"),
    (r"^(d \(mm\)|diameter\s*\(mm\))$", "droplet_diameter", "mm", "Diametro della goccia"),
    (r"^dflame/dodrop", "flame_standoff_ratio", "-", "Rapporto diametro fiamma / diametro iniziale goccia"),
    (r"^(dflame|df)\s*\(?mm\)?", "flame_diameter", "mm", "Diametro della fiamma"),
    (r"^cool flame dia", "cool_flame_diameter", "-", "Diametro della fiamma fredda (cool flame)"),
    (r"^(q\s*\(w\)|q\(w\)|radiance|flame radiance)", "radiance", "W", "Radianza della fiamma"),
    (r"^t_max", "max_temperature", "K", "Temperatura massima della fiamma"),
    (r"^(k\s*\(mm\^?2/s(ec)?\)|kb|kavg)", "burning_rate_constant", "mm^2/s", "Costante di combustione (burning rate) K"),
    (r"^d_tmax/d_drop", "tmax_position_ratio", "-", "Posizione di Tmax / diametro goccia"),
    (r"^d_ohm/d_drop", "oh_max_position_ratio", "-", "Posizione del massimo di OH / diametro goccia"),
    (r"^dext_(cool|cf).*errorbar\+", "extinction_diameter_cool_err_plus", "mm", "Barra d'errore superiore diametro di estinzione cool flame"),
    (r"^dext_(cool|cf).*errorbar-", "extinction_diameter_cool_err_minus", "mm", "Barra d'errore inferiore diametro di estinzione cool flame"),
    (r"^dext_(cool|cf)", "extinction_diameter_cool_flame", "mm", "Diametro di estinzione della fiamma fredda (cool flame)"),
    (r"^dext_(hot|hf)", "extinction_diameter_hot_flame", "mm", "Diametro di estinzione della fiamma calda (hot flame)"),
]

X_PATTERN = re.compile(r"^(t|time)\b|^t/|^(p|pressure)(\s*\(atm\))?$|^do(\s*\(mm\))?$", re.IGNORECASE)
FUEL_CODES = {"7": "n-heptane", "10": "n-decane", "12": "n-dodecane"}
JOURNALS = {"PROCI": "Proceedings of the Combustion Institute", "MST": "Microgravity Science and Technology"}


def base_name(col: str) -> str:
    """Nome colonna senza suffisso di duplicazione pandas (.1, .2) e spazi."""
    return re.sub(r"\.\d+$", "", str(col)).strip()


def is_unnamed(col: str) -> bool:
    return base_name(col).lower().startswith("unnamed")


def quantity(col: str) -> tuple[str, str, str]:
    b = base_name(col)
    low = b.lower()
    for pat, code, unit, desc in QUANTITY_RULES:
        if re.search(pat, low):
            return code, unit, desc
    m = re.search(r"\(([^)]*)\)\s*$", b)
    unit = m.group(1) if m else ""
    code = re.sub(r"[^a-z0-9]+", "_", low).strip("_") or "value"
    return code, unit, b


def dataset_name(source_file: str) -> str:
    stem = re.sub(r"\.(csv|xlsx?|xls)$", "", source_file, flags=re.IGNORECASE)
    stem = re.sub(r"^PSI-\d+_?", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^Analyzed[ _]Data_?", "", stem, flags=re.IGNORECASE)
    return stem or source_file


def publication(dataset: str) -> tuple[str, int | None]:
    m = re.match(r"([A-Za-z]+)_(?:[A-Za-z0-9]+_)?(\d{4})", dataset)
    if not m:
        return "", None
    return JOURNALS.get(m.group(1).upper(), m.group(1)), int(m.group(2))


def _col_kind(s: pd.Series) -> str:
    nn = s.dropna()
    if nn.empty:
        return "empty"
    nums = pd.to_numeric(nn, errors="coerce")
    n_text = int(nums.isna().sum())
    if n_text == 0:
        return "numeric"
    if n_text >= len(nn) * 0.5 or len(nn) <= 5:
        return "label"
    return "numeric"


def _kind_from_tokens(tokens: list[str], current: str) -> str:
    text = " ".join(tokens).lower()
    if re.search(r"\bsim", text):
        return "simulation"
    if re.search(r"\bmodel", text):
        return "model"
    if re.search(r"\bexp", text):
        return "experiment"
    return current


@dataclass
class _Block:
    cols: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    kind: str = "unspecified"


def split_series(df: pd.DataFrame) -> list[_Block]:
    blocks: list[_Block] = []
    context: list[str] = []
    kind = "unspecified"
    last_was_label = False
    blank_run = 0
    current: _Block | None = None

    def close():
        nonlocal current
        if current and len(current.cols) >= 2:
            blocks.append(current)
        current = None

    for col in df.columns:
        k = _col_kind(df[col])
        b = base_name(col)
        if k == "empty":
            close()
            if is_unnamed(col):
                last_was_label = False
                blank_run += 1
                if blank_run >= 2:  # doppio separatore = nuovo gruppo di figure: l'etichetta non si propaga
                    context = []
                continue
            blank_run = 0
            # colonna-titolo (intestazione significativa, nessun valore)
            if not last_was_label:
                context = []
            context.append(b)
            kind = _kind_from_tokens([b], kind)
            last_was_label = True
        elif k == "label":
            blank_run = 0
            close()
            if not last_was_label:
                context = []
            if not is_unnamed(col):
                context.append(b)
            for v in df[col].dropna().astype(str).str.strip():
                if v and v not in context and not v.replace(".", "").isdigit():
                    context.append(v)
            kind = _kind_from_tokens(context, kind)
            last_was_label = True
        else:
            last_was_label = False
            blank_run = 0
            starts_new = (
                current is None
                or (X_PATTERN.search(b) and len(current.cols) >= 2)
                or b == base_name(current.cols[0])
            )
            if starts_new:
                close()
                current = _Block(cols=[], context=list(context), kind=kind)
            current.cols.append(col)
    close()
    return blocks


def _conditions(context: list[str]) -> dict:
    text = " ".join(context)
    out: dict = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*atm", text, re.IGNORECASE)
    if m:
        out["pressure_atm"] = float(m.group(1))
    m = re.search(r"\bdo\s*(\d+(?:\.\d+)?)\s*mm", text, re.IGNORECASE)
    if m:
        out["d0_mm"] = float(m.group(1))
    m = re.search(r"\bC(\d{1,2})\b", text)
    if m and m.group(1) in FUEL_CODES:
        out["fuel"] = FUEL_CODES[m.group(1)]
    return out


def _single_number(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def tidy_table(source_file: str, df: pd.DataFrame, figure: dict | None = None):
    """Restituisce (series_rows, points_df) per una tabella sorgente."""
    figure = figure or {}
    tag = experiment_tag(source_file)
    ds = dataset_name(source_file)
    series_rows, frames = [], []
    blocks = split_series(df)
    has_model = any(b.kind in ("model", "simulation") for b in blocks)
    for i, blk in enumerate(blocks, start=1):
        x_col, y_cols = blk.cols[0], blk.cols[1:]
        cond = _conditions(blk.context)
        fuel = cond.get("fuel") or (figure.get("fuel") if figure.get("fuel") and "," not in str(figure.get("fuel")) else None)
        pressure = cond.get("pressure_atm", _single_number(figure.get("pressure_atm")))
        d0 = cond.get("d0_mm", _single_number(figure.get("d0_mm")))
        label = " · ".join(blk.context) if blk.context else blk.kind
        xq, xu, xd = quantity(x_col)
        for j, y_col in enumerate(y_cols, start=1):
            yq, yu, yd = quantity(y_col)
            kind, inferred = blk.kind, False
            if kind == "unspecified":
                kind = _kind_from_tokens([re.sub(r"[_\-]", " ", base_name(y_col))], kind)
                if kind == "unspecified" and has_model:
                    kind, inferred = "experiment", True
            pts = pd.DataFrame({
                "x": pd.to_numeric(df[x_col], errors="coerce"),
                "y": pd.to_numeric(df[y_col], errors="coerce"),
            }).dropna().reset_index(drop=True)
            if pts.empty:
                continue
            sid = f"{ds}:{i:02d}{'' if len(y_cols) == 1 else chr(96 + j)}"
            meta = {
                "series_id": sid, "experiment_tag": tag, "dataset": ds, "source_file": source_file,
                "series_label": label if label != "unspecified" else kind, "kind": kind,
                "kind_inferred": inferred, "fuel": fuel,
                "pressure_atm": pressure, "d0_mm": d0,
                "x_var": xq, "x_unit": xu, "x_column": str(x_col),
                "y_var": yq, "y_unit": yu, "y_column": str(y_col), "y_description": yd,
            }
            pts.insert(0, "point_idx", range(len(pts)))
            for k, v in reversed(list(meta.items())):
                pts.insert(0, k, v)
            frames.append(pts)
            series_rows.append({
                **meta, "x_description": xd, "n_points": len(pts),
                "x_min": float(pts.x.min()), "x_max": float(pts.x.max()),
                "y_min": float(pts.y.min()), "y_max": float(pts.y.max()),
            })
    points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return series_rows, points


def parse_readme(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Il file ReadMe elenca le figure: combustibile, d0, P, composizione, riferimento DOI."""
    cols = {base_name(c).lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_file, c_ref = pick("data file"), pick("reference")
    if c_file is None:
        return pd.DataFrame()
    fig_col = next((c for c in df.columns if df[c].astype(str).str.match(r"^Fig\.?\s*\d+").any()), None)
    inert_col = pick("xo3", "xhe")
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r[c_file]) or str(r[c_file]).strip().lower() == "data file":
            continue
        rows.append({
            "experiment_tag": experiment_tag(source_file),
            "dataset": str(r[c_file]).strip(),
            "figure": str(r[fig_col]).strip() if fig_col is not None and pd.notna(r[fig_col]) else None,
            "fuel": r.get(pick("fuel")) if pick("fuel") else None,
            "d0_mm": r.get(pick("do (mm)")) if pick("do (mm)") else None,
            "pressure_atm": r.get(pick("p (atm)")) if pick("p (atm)") else None,
            "x_o2": r.get(pick("xo2")) if pick("xo2") else None,
            "x_n2": r.get(pick("xn2")) if pick("xn2") else None,
            "x_third": r.get(inert_col) if inert_col else None,
            "reference_doi": str(r[c_ref]).strip() if c_ref is not None and pd.notna(r[c_ref]) else None,
            "readme_file": source_file,
        })
    out = pd.DataFrame(rows)
    for c in ["fuel", "d0_mm", "pressure_atm", "x_o2", "x_n2", "x_third"]:
        out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip())
    return out
