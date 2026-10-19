"""
liquidity.py  (Entrega #4, Fase 5)

Implementa lo que `restricciones_liquidez.qmd` y
`reglas_salida_regimen_posicion.qmd` dejaron especificado pero no conectado
al motor:

1. Costo de ejecución VARIABLE (Corwin y Schultz, 2012)
   Spread diario estimado con High/Low sin ajustar, variante backward-looking
   (usa t-1 y t), truncado en cero. Como el estimador diario es muy ruidoso
   (muchos ceros y picos), el costo usa la media de las últimas
   `SPREAD_WINDOW` ruedas, conocida al cierre de t-1 (la operación de la fila
   t se ejecuta al cierre de t-1). Costo por lado = fijo (66,55 pb) + medio
   spread. La media de largo plazo coincide, por construcción, con el costo
   constante de la Fase 0; lo nuevo es que el costo SUBE en los días de
   estrés, que es justamente cuando la estrategia opera.

2. Tope de participación de 10% del ADV
   ADV_20 en pesos (Close x Volume, sin ajustar) conocido en t-1. Con el
   monto operado de cada día (motor de comparators.py) y un AUM hipotético
   (500 millones de ARS, como en el .qmd, interpretado como valor de la
   cartera en cada fecha) se cuentan las órdenes que superarían el 10% del
   ADV. Como el AUM es un supuesto arbitrario, se reporta además la
   CAPACIDAD: el valor máximo de cartera (en pesos de la fecha de la orden)
   con el que la orden no superaría el tope. Es un diagnóstico, no se fuerza
   en el backtest.

3. Piso de exposición gatillado por drawdown (reglas de salida, Sección 4)
   Techo_t = 100% si DD_t >= -D_max; si no,
   max(E_min, 100% - kappa * (|DD_t| - D_max)); exposición final =
   min(exposición, techo). DD_t = caída del precio del activo desde su
   máximo de las últimas `lookback` ruedas, hasta t-1. Desactivado por
   defecto; se evalúa como sensibilidad (Fase 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_engine import (
    COST_BPS_PER_SIDE_BY_ASSET,
    COST_BPS_PER_SIDE_BY_ASSET_P95,
    FIXED_COST_BPS_PER_SIDE,
)
from window_selection import EPISODE_CATALOG

SPREAD_WINDOW = 20          # ruedas para suavizar el spread diario
ADV_WINDOW = 20             # ADV_20 (restricciones_liquidez.qmd)
AUM_ARS = 500_000_000       # AUM hipotético (restricciones_liquidez.qmd), parametrizable
ADV_CAP_PCT = 0.10          # tope de participación (mismo .qmd)
COST_MODES = ("medio", "p95", "variable")

DEFAULT_DRAWDOWN_FLOOR = {"d_max": 0.20, "kappa": 2.0, "e_min": 25.0, "lookback": 252}


# --------------------------------------------------------------------------
# 1. Spread de Corwin-Schultz y costo variable
# --------------------------------------------------------------------------

def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Spread bid-ask de Corwin y Schultz (2012), variante backward-looking, truncado en 0."""
    log_hl = np.log(high / low)
    beta = log_hl ** 2 + log_hl.shift(1) ** 2
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def spread_known_at(spread: pd.Series, window: int = SPREAD_WINDOW) -> pd.Series:
    """Media de las últimas `window` ruedas, conocida al cierre de t-1 (fila t)."""
    return spread.rolling(window, min_periods=max(5, window // 2)).mean().shift(1)


def spread_frames(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spread diario (fracción) por activo."""
    return pd.DataFrame({t: corwin_schultz_spread(df["High"], df["Low"]) for t, df in price_data.items()})


def variable_cost_bps(price_data: dict[str, pd.DataFrame], window: int = SPREAD_WINDOW,
                      fixed_bps: float = FIXED_COST_BPS_PER_SIDE) -> pd.DataFrame:
    """Costo por lado (pb) por fecha y activo = fijo + medio spread CS suavizado (conocido en t-1)."""
    sp = spread_frames(price_data)
    return fixed_bps + 0.5 * sp.apply(spread_known_at, window=window) * 10_000


def cost_for_mode(mode: str, price_data: dict | None, window: int = SPREAD_WINDOW):
    """Costo a pasar al motor según el modo. Sin datos High/Low, 'variable' cae a 'medio'."""
    if mode not in COST_MODES:
        raise ValueError(f"Modo de costo desconocido: {mode!r}. Opciones: {COST_MODES}")
    if mode == "p95":
        return dict(COST_BPS_PER_SIDE_BY_ASSET_P95), "p95"
    if mode == "variable" and price_data:
        return variable_cost_bps(price_data, window), "variable"
    return dict(COST_BPS_PER_SIDE_BY_ASSET), "medio" if mode == "medio" else "medio (sin High/Low)"


def spread_by_episode(price_data: dict[str, pd.DataFrame], catalog: dict | None = None,
                      window_days: int = 5) -> pd.DataFrame:
    """
    Validación de sentido del .qmd: spread medio en ±`window_days` ruedas de
    cada episodio contra el spread medio de ese año calendario.
    """
    catalog = catalog or EPISODE_CATALOG
    sp = spread_frames(price_data)
    rows = []
    for name, dates in catalog.items():
        for t in sp.columns:
            s = sp[t].dropna()
            vals = []
            for d in dates:
                pos = s.index.searchsorted(pd.Timestamp(d))
                if pos >= len(s):
                    continue
                vals.append(s.iloc[max(0, pos - window_days): pos + window_days + 1])
            if not vals:
                continue
            ev = pd.concat(vals)
            year = s[s.index.year == pd.Timestamp(dates[0]).year]
            if ev.empty or year.empty:
                continue
            rows.append({"episodio": name, "activo": t, "spread_episodio_pb": 1e4 * ev.mean(),
                         "spread_anual_pb": 1e4 * year.mean(), "razon": ev.mean() / year.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Participación en el ADV y capacidad
# --------------------------------------------------------------------------

def adv_ars(price_data: dict[str, pd.DataFrame], window: int = ADV_WINDOW) -> pd.DataFrame:
    """ADV_20 en pesos (Close x Volume sin ajustar), conocido al cierre de t-1."""
    return pd.DataFrame({
        t: (df["Close"] * df["Volume"]).rolling(window, min_periods=window // 2).mean().shift(1)
        for t, df in price_data.items()
    })


def adv_diagnostics(trades_units: pd.DataFrame, adv: pd.DataFrame, aum_ars: float = AUM_ARS,
                    cap_pct: float = ADV_CAP_PCT, nav_prev: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """
    `trades_units`: monto operado por día y activo en unidades del capital
    inicial (salida de simulate_portfolio(..., return_trades=True)).
    `nav_prev`: NAV de la cartera antes de operar ese día, en las mismas
    unidades (nav.shift(1), con 1 el primer día). Con él, cada orden se
    expresa como FRACCIÓN DE LA CARTERA del día, y el AUM se interpreta como
    el valor de la cartera en esa fecha, en pesos de esa fecha.

    Sin `nav_prev`, el AUM es el capital INICIAL y la cartera crece con el
    NAV. Con un NAV que se multiplicó más de 100 veces en pesos, eso deja al
    final una cartera irreal frente al mercado.

    Devuelve:
      - por_activo y por_anio: órdenes, violaciones del tope con `aum_ars`,
        participación mediana y máxima, y capacidad = valor de cartera
        máximo (pesos de la fecha de la orden) con el que la orden respeta
        el tope (percentil 5 y mínimo).
      - participacion: participación diaria por activo.
    """
    tr = trades_units.abs()
    if nav_prev is not None:
        tr = tr.div(nav_prev.reindex(tr.index), axis=0)
    a = adv.reindex(index=tr.index, columns=tr.columns)
    traded = tr > 1e-9
    part = (tr * aum_ars / a).where(traded)
    capacity = (cap_pct * a / tr).where(traded)

    def _summ(p, cap):
        p = p.stack().dropna()
        cap = cap.stack().dropna()
        return {"ordenes": int(len(p)), "violaciones": int((p > cap_pct).sum()),
                "pct_violaciones": float((p > cap_pct).mean()) if len(p) else np.nan,
                "participacion_mediana": float(p.median()) if len(p) else np.nan,
                "participacion_max": float(p.max()) if len(p) else np.nan,
                "capacidad_p5_ARS": float(cap.quantile(0.05)) if len(cap) else np.nan,
                "capacidad_min_ARS": float(cap.min()) if len(cap) else np.nan}

    por_activo = pd.DataFrame({t: _summ(part[[t]], capacity[[t]]) for t in tr.columns}).T
    por_anio = pd.DataFrame({y: _summ(part[part.index.year == y], capacity[capacity.index.year == y])
                             for y in sorted(set(tr.index.year))}).T
    return {"por_activo": por_activo, "por_anio": por_anio, "participacion": part}


# --------------------------------------------------------------------------
# 3. Piso de exposición por drawdown
# --------------------------------------------------------------------------

def drawdown_ceiling(log_returns: pd.Series, d_max: float, kappa: float, e_min: float,
                     lookback: int = 252) -> pd.Series:
    """Techo de exposición (%) por drawdown del precio, conocido en t-1."""
    price = np.exp(log_returns.fillna(0).cumsum())
    peak = price.rolling(lookback, min_periods=1).max()
    dd = (price / peak - 1).shift(1)
    excess = (-dd - d_max).clip(lower=0)
    ceiling = (100 - kappa * 100 * excess).clip(lower=e_min)
    return ceiling.where(dd.notna(), 100.0)


def apply_drawdown_floor(exposures: dict[str, pd.Series], returns_by_asset: dict[str, pd.Series],
                         d_max: float = 0.20, kappa: float = 2.0, e_min: float = 25.0,
                         lookback: int = 252, hysteresis_pp: float | None = None) -> dict[str, pd.Series]:
    """
    Exposición final = min(exposición de la señal, techo por drawdown), con la
    misma banda de histéresis de la señal (5 p.p.). Sin la banda, el techo se
    mueve un poco cada día que el precio sigue en drawdown y la cartera
    operaría a diario pagando costos por cambios mínimos. Con la banda, la
    exposición puede quedar hasta 5 p.p. por encima del techo.
    """
    from signal_generator import HYSTERESIS_PP, apply_hysteresis
    band = HYSTERESIS_PP if hysteresis_pp is None else hysteresis_pp
    out = {}
    for t, e in exposures.items():
        ceil = drawdown_ceiling(returns_by_asset[t], d_max, kappa, e_min, lookback).reindex(e.index)
        capped = e.where(e.isna(), np.minimum(e, ceil))
        out[t] = apply_hysteresis(capped, band_pp=band).where(e.notna()) if band > 0 else capped
    return out


# --------------------------------------------------------------------------
# Sensibilidad al costo
# --------------------------------------------------------------------------

def cost_sensitivity(result: dict, price_data: dict | None,
                     names=("Estrategia", "EW 100%", "Exposición fija (media ex post)")) -> pd.DataFrame:
    """Mismas carteras con costo medio, p95 y variable."""
    from comparators import _simple_returns_frame, build_comparator_exposures, simulate_portfolio
    from performance_metrics import summary

    exposures = build_comparator_exposures(result)
    rets = _simple_returns_frame(result, exposures["Estrategia"].columns)
    rows = []
    rf = result.get("cash_rate")
    for mode in COST_MODES:
        cost, used = cost_for_mode(mode, price_data, result["config"].spread_window)
        for n in names:
            df = simulate_portfolio(rets, exposures[n], cost, result["config"].rebalance, cash_rate=rf)
            s = summary(df["net_return"], n, rf)
            rows.append({"costo": used, "cartera": n, "retorno_anual": s["Retorno anualizado"],
                         "sharpe": s["Sharpe"], "max_drawdown": s["Max Drawdown"],
                         "costo_total_pp": 100 * df["cost"].sum()})
    return pd.DataFrame(rows)
