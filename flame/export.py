"""Esportazione DataFrame -> CSV / Excel con foglio di metadati e fonti."""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone

import pandas as pd


def safe_filename(title: str, ext: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", title or "dati").strip("_")[:60] or "dati"
    return f"{stem}.{ext}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM: Excel apre correttamente gli accenti


def to_xlsx_bytes(df: pd.DataFrame, title: str = "dati", meta: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="dati", index=False)
        ws = xw.sheets["dati"]
        for i, col in enumerate(df.columns, start=1):
            width = min(45, max(10, len(str(col)) + 2, *(len(str(v)) for v in df[col].head(200))))
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
        ws.freeze_panes = "A2"
        rows = [("Titolo", title), ("Generato (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
                ("Origine", "Flame in Freefall — NASA Physical Science Informatics (PSI)"),
                ("Righe", len(df))]
        rows += [(k, ", ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v)) for k, v in (meta or {}).items()]
        pd.DataFrame(rows, columns=["campo", "valore"]).to_excel(xw, sheet_name="metadati", index=False)
        xw.sheets["metadati"].column_dimensions["A"].width = 22
        xw.sheets["metadati"].column_dimensions["B"].width = 110
    return buf.getvalue()


def sources_of(df: pd.DataFrame) -> list[str]:
    """File sorgente presenti nel risultato (per la citazione dei dati)."""
    for col in ("source_file", "file_name"):
        if col in df.columns:
            return sorted(df[col].dropna().astype(str).unique().tolist())
    return []
