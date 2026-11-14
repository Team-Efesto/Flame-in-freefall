"""Costruzione grafici Plotly coerenti con il tema scuro dell'app.

Regole (dataviz): palette categoriale a ordine fisso (mai ciclata, colore legato all'entita'),
un solo asse y (grandezze diverse -> pannelli separati), linee 2px, marker >= 8px,
griglia recessiva, tooltip sempre attivo, legenda per >= 2 serie.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Palette categoriale di riferimento, varianti per superficie scura (validate CVD)
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
TEXT = "#E8ECF4"
MUTED = "#A9B3C9"
GRID = "rgba(169,179,201,0.14)"
MAX_COLORS = len(PALETTE)

CHART_TYPES = ("line", "scatter", "line+markers", "bar", "box", "histogram")


def style(fig: go.Figure, title: str | None = None, height: int = 460) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        title=dict(text=title or "", font=dict(size=16, color=TEXT), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, title_text="", font=dict(color=TEXT)),
        margin=dict(l=10, r=10, t=56 if title else 24, b=10),
        hoverlabel=dict(bgcolor="#141A2B", bordercolor="#2E3856", font=dict(color=TEXT)),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(color=MUTED))
    return fig


def color_map(values) -> dict:
    """Colore stabile per entita' (ordine di prima apparizione); oltre 8 -> grigio 'altro'."""
    uniq = list(dict.fromkeys(str(v) for v in values))
    return {v: (PALETTE[i] if i < MAX_COLORS else "#6B7280") for i, v in enumerate(uniq)}


def build_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None = None, color: str | None = None,
                facet: str | None = None, title: str | None = None, x_label: str | None = None,
                y_label: str | None = None, log_x: bool = False, log_y: bool = False,
                error_y: str | None = None) -> go.Figure:
    for c in (x, y, color, facet, error_y):
        if c and c not in df.columns:
            raise ValueError(f"Colonna '{c}' non presente. Colonne disponibili: {list(df.columns)}")
    if chart_type not in CHART_TYPES:
        raise ValueError(f"chart_type deve essere uno di {CHART_TYPES}")
    if y is None and chart_type not in ("histogram",):
        raise ValueError("y e' obbligatorio per questo tipo di grafico")

    d = df.copy()
    if color:
        d[color] = d[color].astype(str)
        n = d[color].nunique()
        if n > MAX_COLORS:
            raise ValueError(f"'{color}' ha {n} categorie: massimo {MAX_COLORS} colori distinguibili. "
                             "Filtra le serie, aggrega, oppure usa `facet` per pannelli separati.")
    cmap = color_map(d[color]) if color else None
    labels = {k: v for k, v in ((x, x_label), (y, y_label)) if k and v}
    common = dict(color=color, color_discrete_map=cmap, labels=labels, log_x=log_x)
    if facet:
        common.update(facet_col=facet, facet_col_wrap=2)
    if chart_type != "histogram":
        common["log_y"] = log_y
    if chart_type in ("line", "line+markers"):
        d = d.sort_values([c for c in (color, facet, x) if c])
        fig = px.line(d, x=x, y=y, markers=(chart_type == "line+markers"), error_y=error_y, **common)
        fig.update_traces(line=dict(width=2), marker=dict(size=8))
    elif chart_type == "scatter":
        fig = px.scatter(d, x=x, y=y, error_y=error_y, **common)
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color="#0B0F1A")))
    elif chart_type == "bar":
        fig = px.bar(d, x=x, y=y, error_y=error_y, barmode="group", **common)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.25, bargroupgap=0.06)
    elif chart_type == "box":
        fig = px.box(d, x=x, y=y, points="outliers", **common)
    else:  # histogram
        common.pop("log_y", None)
        fig = px.histogram(d, x=x, nbins=40, barmode="overlay", **common)
        fig.update_traces(opacity=0.8)
    if not color and chart_type != "box":
        fig.update_traces(marker_color=PALETTE[0], selector=dict(type="bar"))
        fig.update_traces(line_color=PALETTE[0], marker_color=PALETTE[0], selector=dict(mode="lines"))
        fig.update_traces(line_color=PALETTE[0], marker_color=PALETTE[0], selector=dict(mode="lines+markers"))
        fig.update_traces(marker_color=PALETTE[0], selector=dict(mode="markers"))
    if facet:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(color=TEXT)))
        fig.update_yaxes(matches=None, showticklabels=True)
    n_traces = len({t.name for t in fig.data})
    fig.update_layout(showlegend=n_traces >= 2)
    return style(fig, title, height=460 if not facet else 380 * ((d[facet].nunique() + 1) // 2))


def series_chart(points: pd.DataFrame, log_x: bool = False, log_y: bool = False, title: str | None = None) -> go.Figure:
    """Grafico delle serie selezionate in dashboard: pannelli separati se le grandezze y differiscono."""
    d = points.copy()
    d["serie"] = d["series_id"].str.split(":").str[-1] + " · " + d["series_label"].astype(str).str.slice(0, 40)
    facet = "y_var" if d["y_var"].nunique() > 1 else None
    x_unit = d["x_unit"].iloc[0]
    y_unit = d["y_unit"].iloc[0] if not facet else ""
    x_label = f"{d['x_var'].iloc[0]} [{x_unit}]" if d["x_var"].nunique() == 1 else "x"
    y_label = f"{d['y_var'].iloc[0]} [{y_unit}]" if not facet else "y"
    mode = "line+markers" if d.groupby("series_id").size().max() <= 60 else "line"
    return build_chart(d, mode, "x", "y", color="serie", facet=facet, title=title,
                       x_label=x_label, y_label=y_label, log_x=log_x, log_y=log_y)
