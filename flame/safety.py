"""Confronto dei risultati di simulazione con i limiti di sicurezza antincendio spaziale.

I limiti sono estratti dai PDF in data_nasa_raw/safety_pdf da build_safety_db.py e salvati in
data/safety/safety_limits.parquet, in UNITA' CANONICHE (vedi CANONICAL_UNITS). Ogni limite ha la
citazione verificata (file + pagina + frase originale).

Uso tipico:

    from flame.safety import SafetyLimits
    lim = SafetyLimits()

    # 1) valori singoli o serie con grandezza e unita' esplicite
    lim.check([1800, 2600], "max_temperature", "K")

    # 2) DataFrame di output di una simulazione: le colonne vengono riconosciute dal nome
    #    (es. "temperature_K", "O2 [%]", "x_o2", "air_velocity_m_s", "flame_length_mm")
    lim.check_dataframe(sim_df)
    lim.check_dataframe(sim_df, mapping={"T": ("temperature", "K")})   # mapping esplicito

    # 3) tutte le serie di simulazione gia' presenti nel DataStore dell'app
    lim.check_store(store, kinds=("simulation", "model"))

Esito per ogni (grandezza, limite) - colonna `violation` (bool) + `esito` leggibile:
    limiti "compliant_zone": "conforme" / "non conforme"
    limiti "hazard_zone":    "fuori zona di rischio" / "in zona di rischio"
    (vedi n_violations, worst_value, margin per i dettagli)
Di default sono esclusi i limiti ad ambito ristretto (applies_by_default=False, es. volumi inertizzati,
capacita' di maschere/coperte): includerli con include_all=True o limit_ids=[...].
Il margine e' la distanza (in unita' canoniche) del valore peggiore dal bordo del limite:
positivo = dentro la zona sicura, negativo = violazione.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import SAFETY_DIR


def F_TO_K(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0 + 273.15


# Grandezza canonica -> unita' canonica (quella in cui sono salvati i limiti)
CANONICAL_UNITS: dict[str, str] = {
    "oxygen_concentration": "%",      # frazione molare/volumetrica in percentuale
    "air_velocity": "cm/s",
    "burn_length": "cm",              # lunghezza di propagazione della fiamma sul campione
    "material_separation": "cm",
    "temperature": "K",
    "ammonia_concentration": "ppm",
    "exposure_time": "min",
    "pressure": "kPa",
}

# Conversioni verso l'unita' canonica: unita' (minuscolo, normalizzata) -> funzione
_LIN = lambda k: (lambda v: v * k)  # noqa: E731
UNIT_CONVERSIONS: dict[str, dict[str, callable]] = {
    "oxygen_concentration": {"%": _LIN(1), "percent": _LIN(1), "pct": _LIN(1), "vol%": _LIN(1), "mol%": _LIN(1),
                             "fraction": _LIN(100), "mole fraction": _LIN(100), "mol/mol": _LIN(100), "-": _LIN(100),
                             "": _LIN(100), "ppm": _LIN(1e-4)},
    "air_velocity": {"cm/s": _LIN(1), "mm/s": _LIN(0.1), "m/s": _LIN(100), "ft/min": _LIN(0.508),
                     "ft/s": _LIN(30.48), "km/h": _LIN(100 / 3.6)},
    "burn_length": {"cm": _LIN(1), "mm": _LIN(0.1), "m": _LIN(100), "in": _LIN(2.54), "ft": _LIN(30.48)},
    "material_separation": {"cm": _LIN(1), "mm": _LIN(0.1), "m": _LIN(100), "in": _LIN(2.54)},
    "temperature": {"k": _LIN(1), "c": lambda v: v + 273.15, "°c": lambda v: v + 273.15, "degc": lambda v: v + 273.15,
                    "f": F_TO_K, "°f": F_TO_K, "degf": F_TO_K},
    "ammonia_concentration": {"ppm": _LIN(1), "ppmv": _LIN(1), "%": _LIN(1e4), "percent": _LIN(1e4), "pct": _LIN(1e4),
                              "fraction": _LIN(1e6), "-": _LIN(1e6)},
    "exposure_time": {"min": _LIN(1), "s": _LIN(1 / 60), "sec": _LIN(1 / 60), "h": _LIN(60), "hr": _LIN(60)},
    "pressure": {"kpa": _LIN(1), "pa": _LIN(1e-3), "atm": _LIN(101.325), "bar": _LIN(100), "psi": _LIN(6.894757),
                 "psia": _LIN(6.894757), "mmhg": _LIN(0.133322), "torr": _LIN(0.133322)},
}

# Nomi di variabili/colonne (gia' "snake_case", senza unita') -> grandezza canonica.
# Riconoscimento STRETTO: solo grandezze fisicamente equivalenti al limite.
ALIASES: list[tuple[str, str]] = [
    (r"^(x_?o2|y_?o2|o2|o2_conc(entration)?|oxygen(_conc(entration)?)?|oxygen_(mole|volume)_fraction|"
     r"xo2_before_test|o2_(fraction|percent|pct))$", "oxygen_concentration"),
    (r"^(air_|flow_|gas_|ventilation_|forced_flow_)?(velocity|speed)$|^u_?(inf|air|flow)$", "air_velocity"),
    (r"^(burn|burned|flame_spread|spread|propagation)_(length|distance)$", "burn_length"),
    (r"^(material_)?(separation|gap|spacing)(_distance)?$", "material_separation"),
    (r"^(max_|peak_|flame_|gas_|adiabatic_flame_)?temperature$|^t_?max$|^t_?(flame|gas)$", "temperature"),
    (r"^(nh3|ammonia)(_conc(entration)?)?$", "ammonia_concentration"),
    (r"^exposure(_time|_duration)?$", "exposure_time"),
    (r"^(ambient_|cabin_|total_)?pressure$|^p_?(amb|cabin|tot)$", "pressure"),
]
# Confronti solo INDICATIVI (grandezza diversa ma usata come proxy): attivi con allow_proxy=True
PROXY_ALIASES: list[tuple[str, str]] = [
    (r"^(flame_diameter|flame_length|flame_height|flame_standoff)$", "burn_length"),
]

# Suffissi di unita' nei nomi colonna: "temperature_K", "o2_pct", "velocity_m_s", "x [cm/s]", "T (°C)"
_SUFFIX_UNITS = {"k": "k", "c": "c", "degc": "c", "f": "f", "degf": "f", "pct": "%", "percent": "%",
                 "cm_s": "cm/s", "mm_s": "mm/s", "m_s": "m/s", "mm": "mm", "cm": "cm", "m": "m", "in": "in",
                 "ppm": "ppm", "ppmv": "ppmv", "s": "s", "min": "min", "h": "h", "kpa": "kpa", "pa": "pa",
                 "atm": "atm", "psi": "psi", "psia": "psia", "mmhg": "mmhg", "bar": "bar"}


def _snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _norm_unit(unit: str | None) -> str:
    u = (unit or "").strip().lower().replace("º", "°").replace(" ", "")
    return {"kelvin": "k", "celsius": "c", "fahrenheit": "f", "%vol": "%", "vol.%": "%"}.get(u, u)


def split_name_unit(column: str) -> tuple[str, str | None]:
    """'temperature_K' -> ('temperature', 'k'); 'O2 [%]' -> ('o2', '%'); 'velocity' -> ('velocity', None)."""
    m = re.match(r"^\s*(.*?)\s*[\[(]\s*([^\])]+?)\s*[\])]\s*$", column)
    if m:
        return _snake(m.group(1)), _norm_unit(m.group(2))
    s = _snake(column)
    for suf in sorted(_SUFFIX_UNITS, key=len, reverse=True):
        if s.endswith("_" + suf) and len(s) > len(suf) + 1:
            return s[: -len(suf) - 1], _SUFFIX_UNITS[suf]
    return s, None


def canonical_parameter(name: str, allow_proxy: bool = False) -> tuple[str | None, bool]:
    """Nome variabile -> (grandezza canonica | None, is_proxy)."""
    s = _snake(name)
    for rx, p in ALIASES:
        if re.match(rx, s):
            return p, False
    if allow_proxy:
        for rx, p in PROXY_ALIASES:
            if re.match(rx, s):
                return p, True
    return (s, False) if s in CANONICAL_UNITS else (None, False)


def to_canonical(values, parameter: str, unit: str | None) -> tuple[np.ndarray, str]:
    """Converte i valori nell'unita' canonica. Ritorna (valori, nota sull'unita' usata)."""
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    convs = UNIT_CONVERSIONS[parameter]
    u = _norm_unit(unit)
    note = ""
    if parameter == "oxygen_concentration" and u in ("", "-"):
        finite = v[np.isfinite(v)]
        if finite.size and finite.max() > 1.0:   # gia' in percentuale
            u, note = "%", "unita' assente: valori > 1 interpretati come %"
        else:
            note = "unita' assente: valori <= 1 interpretati come frazione molare"
    if u not in convs:
        if u == "" and _norm_unit(CANONICAL_UNITS[parameter]) in convs:
            u, note = _norm_unit(CANONICAL_UNITS[parameter]), f"unita' assente: assunta {CANONICAL_UNITS[parameter]}"
        else:
            raise ValueError(f"Unita' '{unit}' non convertibile per {parameter} (ammesse: {sorted(convs)})")
    return convs[u](v), note


class SafetyLimits:
    def __init__(self, safety_dir=SAFETY_DIR):
        path = safety_dir / "safety_limits.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} non trovato: esegui prima `python build_safety_db.py`")
        self.limits = pd.read_parquet(path)
        req = safety_dir / "safety_requirements.parquet"
        self.requirements = pd.read_parquet(req) if req.exists() else pd.DataFrame()

    # ------------------------------------------------------------------
    def for_parameter(self, parameter: str, limit_types: Iterable[str] | None = None,
                      limit_ids: Iterable[str] | None = None, include_all: bool = False) -> pd.DataFrame:
        g = self.limits[self.limits.parameter == parameter]
        if limit_ids:
            return g[g.limit_id.isin(list(limit_ids))]
        if not include_all and "applies_by_default" in g:
            g = g[g.applies_by_default.astype(bool)]
        if limit_types:
            g = g[g.limit_type.isin(list(limit_types))]
        return g

    @staticmethod
    def _inside(v: np.ndarray, r) -> np.ndarray:
        ok = np.ones_like(v, dtype=bool)
        if not math.isnan(r.lower):
            ok &= (v >= r.lower) if r.lower_inclusive else (v > r.lower)
        if not math.isnan(r.upper):
            ok &= (v <= r.upper) if r.upper_inclusive else (v < r.upper)
        return ok

    @staticmethod
    def _margin(v: np.ndarray, r) -> np.ndarray:
        """Distanza firmata dal bordo: > 0 lato sicuro, < 0 violazione."""
        lo = v - r.lower if not math.isnan(r.lower) else np.full_like(v, np.inf)
        hi = r.upper - v if not math.isnan(r.upper) else np.full_like(v, np.inf)
        inside = np.minimum(lo, hi)                      # > 0 dentro l'intervallo
        return inside if r.bound_type == "compliant_zone" else -inside

    def check(self, values, quantity: str, unit: str | None = None, label: str = "",
              limit_types: Iterable[str] | None = None, allow_proxy: bool = False,
              limit_ids: Iterable[str] | None = None, include_all: bool = False) -> pd.DataFrame:
        """Confronta una sequenza di valori con i limiti applicabili alla grandezza.

        Di default esclude i limiti ad ambito ristretto (applies_by_default=False): usa include_all=True
        oppure limit_ids=["SAF-006", ...] per includerli."""
        param, is_proxy = canonical_parameter(quantity, allow_proxy)
        if param is None or param not in CANONICAL_UNITS:
            return pd.DataFrame()
        v, unit_note = to_canonical(values, param, unit)
        v = v[np.isfinite(v)]
        rows = []
        for r in self.for_parameter(param, limit_types, limit_ids, include_all).itertuples():
            if v.size == 0:
                continue
            inside = self._inside(v, r)
            viol = ~inside if r.bound_type == "compliant_zone" else inside
            margin = self._margin(v, r)
            worst = int(np.argmin(margin))
            rows.append({
                "label": label, "quantity": quantity, "parameter": param, "unit": r.unit,
                "limit_id": r.limit_id, "limit_type": r.limit_type, "bound_type": r.bound_type,
                "limit": _fmt_bounds(r), "scope": r.scope,
                "n_values": int(v.size), "value_min": float(v.min()), "value_max": float(v.max()),
                "n_violations": int(viol.sum()), "pct_violations": round(100 * float(viol.mean()), 2),
                "worst_value": float(v[worst]), "margin": float(margin[worst]),
                "violation": bool(viol.any()),
                "esito": _esito(r.bound_type, bool(viol.any())),
                "confronto": "indicativo (proxy)" if is_proxy else "diretto",
                "unit_note": unit_note, "description": r.description, "citation": r.citation,
            })
        return pd.DataFrame(rows)

    def check_dataframe(self, df: pd.DataFrame, mapping: dict[str, tuple[str, str | None]] | None = None,
                        label: str = "", allow_proxy: bool = False, **kw) -> pd.DataFrame:
        """Riconosce le colonne del DataFrame (nome + unita' nel nome) o usa `mapping` {colonna: (grandezza, unita')}."""
        mapping = dict(mapping or {})
        for col in df.columns:
            if col in mapping:
                continue
            name, unit = split_name_unit(str(col))
            param, _ = canonical_parameter(name, allow_proxy)
            if param:
                mapping[col] = (name, unit)
        out = [self.check(df[c], q, u, label=label or str(c), allow_proxy=allow_proxy, **kw)
               for c, (q, u) in mapping.items()]
        out = [o.assign(column=c) for o, c in zip(out, mapping) if not o.empty]
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    @staticmethod
    def is_long_format(df: pd.DataFrame) -> bool:
        """True per tabelle nel formato di `points` (colonne y + y_var [+ y_unit, x, x_var, x_unit])."""
        return {"y", "y_var"}.issubset(df.columns)

    def check_points(self, df: pd.DataFrame, allow_proxy: bool = False, **kw) -> pd.DataFrame:
        """Verifica una tabella in formato lungo (come `points` o un risultato SQL che ne conserva le colonne):
        raggruppa per serie e grandezza e controlla sia y (y_var/y_unit) sia x (x_var/x_unit)."""
        key = "series_id" if "series_id" in df.columns else None
        out = []
        for axis in ("y", "x"):
            var, unit = f"{axis}_var", f"{axis}_unit"
            if axis not in df.columns or var not in df.columns:
                continue
            cols = [c for c in (key, var, unit) if c and c in df.columns]
            for vals, g in df.groupby(cols, dropna=False):
                vals = dict(zip(cols, vals if isinstance(vals, tuple) else (vals,)))
                u = vals.get(unit)
                r = self.check(g[axis], str(vals[var]), None if pd.isna(u) else str(u),
                               label=str(vals.get(key, vals[var])), allow_proxy=allow_proxy, **kw)
                if not r.empty:
                    out.append(r.assign(series_id=vals.get(key), source=f"colonna {axis} ({vals[var]})"))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def check_any(self, df: pd.DataFrame, **kw) -> pd.DataFrame:
        """Formato lungo -> check_points; altrimenti riconoscimento delle colonne con check_dataframe."""
        return self.check_points(df, **kw) if self.is_long_format(df) else self.check_dataframe(df, **kw)

    def catalog_text(self) -> str:
        """Elenco compatto dei limiti (per il prompt dell'agente)."""
        lines = []
        for r in self.limits.itertuples():
            opt = "" if getattr(r, "applies_by_default", True) else " [opzionale: solo con limit_ids/include_all]"
            lines.append(f"- {r.limit_id} | {r.parameter} | {_fmt_bounds(r)} | {r.limit_type} | {r.scope}{opt} | {r.citation}")
        return "\n".join(lines)

    def check_store(self, store, kinds: Iterable[str] = ("simulation", "model"), allow_proxy: bool = False,
                    **kw) -> pd.DataFrame:
        """Verifica le serie del DataStore (points/series/datasets) dei tipi indicati.

        Controlla y_var e x_var di ogni serie e le condizioni di prova del dataset (x_o2, pressione)."""
        series = store.series[store.series.kind.isin(list(kinds))]
        pts = store.points[store.points.series_id.isin(series.series_id)]
        ds = store.datasets.set_index("dataset") if "dataset" in store.datasets else pd.DataFrame()
        out = []
        for s in series.itertuples():
            p = pts[pts.series_id == s.series_id]
            for axis, var, unit in (("y", s.y_var, s.y_unit), ("x", s.x_var, s.x_unit)):
                r = self.check(p[axis], var, unit, label=s.series_id, allow_proxy=allow_proxy, **kw)
                if not r.empty:
                    out.append(r.assign(series_id=s.series_id, source="serie", experiment_tag=s.experiment_tag))
        for name in series.dataset.unique():  # condizioni ambientali, una volta per dataset
            if name not in ds.index:
                continue
            d = ds.loc[name]
            ids = ",".join(series[series.dataset == name].series_id)
            for col, q, u in (("x_o2", "x_o2", "fraction"), ("pressure_atm", "pressure", "atm")):
                vals = _parse_numbers(d.get(col))
                if vals:
                    r = self.check(vals, q, u, label=name, **kw)
                    if not r.empty:
                        out.append(r.assign(series_id=ids, source=f"condizione dataset ({col})",
                                            experiment_tag=d.get("experiment_tag")))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    @staticmethod
    def summary(results: pd.DataFrame) -> pd.DataFrame:
        """Una riga per limite: quante serie/colonne verificate e quante non conformi."""
        if results.empty:
            return results
        return (results.groupby(["limit_id", "parameter", "limit", "limit_type", "scope"], as_index=False)
                .agg(n_checked=("label", "count"), n_violazioni=("violation", "sum"),
                     worst_margin=("margin", "min"), citation=("citation", "first")))


def _esito(bound_type: str, violation: bool) -> str:
    if bound_type == "hazard_zone":
        return "in zona di rischio" if violation else "fuori zona di rischio"
    return "non conforme" if violation else "conforme"


def _fmt_bounds(r) -> str:
    lo = "" if math.isnan(r.lower) else f"{r.lower:g}"
    hi = "" if math.isnan(r.upper) else f"{r.upper:g}"
    if lo and hi:
        rng = f"{lo}–{hi} {r.unit}"
    elif hi:
        rng = f"{'≤' if r.upper_inclusive else '<'} {hi} {r.unit}"
    else:
        rng = f"{'≥' if r.lower_inclusive else '>'} {lo} {r.unit}"
    return ("zona conforme: " if r.bound_type == "compliant_zone" else "zona di rischio: ") + rng


def _parse_numbers(v) -> list[float]:
    """'0.5 - 1.5' -> [0.5, 1.5]; '0.21' -> [0.21]; None -> []."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(v))]


format_limit = _fmt_bounds  # etichetta leggibile di un limite (riga di safety_limits)
