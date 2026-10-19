"""
comparators.py

Motor de cartera con tenencias explícitas y comparadores de la estrategia.

Este módulo simula la cartera con tenencias en pesos y caja, de modo que:

  - las posiciones derivan con los precios entre rebalanceos;
  - cada rebalanceo (entre activos o dentro de un activo, por cambio de
    exposición) paga el costo por lado sobre el monto operado;
  - la compra inicial se cobra igual a todos;
  - la frecuencia de rebalanceo entre activos es un supuesto explícito
    ("diario", "mensual", "trimestral", "nunca").

La misma función (`simulate_portfolio`) se usa para la estrategia y para
todos los comparadores: la única diferencia entre ellos es la matriz de
exposición objetivo por activo.

Estructura de la cartera
------------------------
Cada activo es un "sleeve" con valor V_i = A_i + K_i
(A_i: posición en el activo; K_i: caja del "sleeve", tasa 0).

  - Día de rebalanceo (según la frecuencia, o cuando cambia la composición):
    V_i = NAV / N para cada activo y A_i = e_i * V_i.
    Mover caja entre "sleeves" no cuesta; comprar o vender el activo sí.
  - Otro día: si la exposición objetivo e_i cambió, A_i = e_i * V_i (solo en
    ese activo). Si no cambió, no se opera y la posición deriva con el precio.

La exposición e_i del día t se decide con información hasta t-1 y se aplica
al retorno de t.

Comparadores (`build_comparator_exposures`)
-------------------------------------------
Todos usan los mismos activos y las mismas fechas que la estrategia:

  - "EW 100%": el benchmark equal-weight, ahora con compra inicial y
    rebalanceo con costos.
  - "Exposición fija": cada activo a la exposición media que tuvo la
    estrategia. Es un diagnóstico ex post (usa la media de todo el período),
    no una regla implementable: separa el efecto de "estar menos invertido"
    del efecto de "cuándo" se baja la exposición.
  - "Vol targeting": e_t = min(1, sigma*_t / sigma_t), con sigma_t = desvío de
    los últimos 60 retornos hasta t-1 y sigma*_t = mediana histórica
    (expanding) de sigma hasta t-1. Misma banda de histéresis de 5 p.p.
  - "MA200": 100% si el precio de t-1 está sobre su media de 200 ruedas,
    0% si no.
  - "Merval": índice S&P Merval, sin costos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_engine import (
    COST_BPS_PER_SIDE_BY_ASSET,
    DEFAULT_COST_BPS_PER_SIDE,
    log_to_simple,
)
from performance_metrics import summary
from signal_generator import HYSTERESIS_PP, apply_hysteresis

REBALANCE_RULES = ("diario", "mensual", "trimestral", "nunca")
DEFAULT_REBALANCE = "mensual"

VOL_TARGET_LOOKBACK = 60      # ruedas para sigma_t
VOL_TARGET_MIN_HISTORY = 252  # ruedas mínimas para la mediana histórica
MA_WINDOW = 200

PORTFOLIO_COLUMNS = ["net_return", "gross_return", "cost", "turnover",
                     "exposure", "n_assets", "nav"]


# --------------------------------------------------------------------------
# Motor
# --------------------------------------------------------------------------

def _is_rebalance_day(rule: str, t: pd.Timestamp, prev: pd.Timestamp | None) -> bool:
    if prev is None:
        return True
    if rule == "diario":
        return True
    if rule == "mensual":
        return (t.year, t.month) != (prev.year, prev.month)
    if rule == "trimestral":
        return (t.year, (t.month - 1) // 3) != (prev.year, (prev.month - 1) // 3)
    if rule == "nunca":
        return False
    raise ValueError(f"Regla de rebalanceo desconocida: {rule!r}. Opciones: {REBALANCE_RULES}")


def _size_after_cost(e: np.ndarray, V: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Posición A' tal que, después de pagar el costo de llegar a ella, la
    fracción invertida del "sleeve" sea exactamente e:
        A' = e * (V - c * |A' - A|).
    Compra (A' >= A): A' = e (V + c A) / (1 + e c).
    Venta  (A' <  A): A' = e (V - c A) / (1 - e c).
    Evita que el costo deje caja negativa.
    """
    buy = e * (V + c * A) / (1 + e * c)
    sell = e * (V - c * A) / (1 - e * c)
    return np.where(buy >= A, buy, sell)


def simulate_portfolio(
    simple_returns: pd.DataFrame,
    target_exposure: pd.DataFrame,
    cost_bps_per_side: dict | float | pd.DataFrame = COST_BPS_PER_SIDE_BY_ASSET,
    rebalance: str = DEFAULT_REBALANCE,
    return_trades: bool = False,
    cash_rate: pd.Series | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simula una cartera equal-weight de "sleeves" con tenencias explícitas.

    Parámetros:
    ----------
    simple_returns : DataFrame fechas x activos, retornos simples diarios.
    target_exposure : DataFrame fechas x activos, exposición objetivo en
        fracción [0, 1]. NaN = activo fuera de la cartera ese día.
    cost_bps_per_side : costo por lado (pb): único, por activo (dict) o
        variable por fecha y activo (DataFrame, Fase 5: fijo + medio spread
        de Corwin-Schultz conocido en t-1). Faltantes -> media del activo.
    return_trades : si True, devuelve además un DataFrame fechas x activos
        con el monto operado (con signo) en unidades del capital inicial
        (1 = capital inicial). Multiplicado por el AUM (Assets Under Management) inicial en pesos da
        el monto de la orden, para el chequeo de ADV (Average Daily Volume).
    rebalance : frecuencia del rebalanceo entre activos.
    cash_rate : Serie (fecha) con el rendimiento simple diario de la caja,
        tasa vigente al cierre de t-1 (macro_data.daily_cash_return). La caja
        de cada "sleeve" lo devenga todos los días. None = caja al 0%.

    Retornos:
    -------
    DataFrame con PORTFOLIO_COLUMNS, indexado por las fechas con al menos un
    activo. `cost` y `turnover` están en fracción del NAV (Net Asset Value) del inicio del día;
    `exposure` es la fracción del NAV invertida al inicio del día, después
    de operar y pagar costos.
    """
    if rebalance not in REBALANCE_RULES:
        raise ValueError(f"Regla de rebalanceo desconocida: {rebalance!r}. Opciones: {REBALANCE_RULES}")
    target = target_exposure.astype(float)
    active_any = target.notna().any(axis=1)
    target = target[active_any]
    if target.empty:
        return pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    assets = list(target.columns)
    rets = simple_returns.reindex(index=target.index, columns=assets).astype(float)

    if isinstance(cost_bps_per_side, pd.DataFrame):
        cm = cost_bps_per_side.reindex(index=target.index, columns=assets)
        cm = cm.fillna(cost_bps_per_side.reindex(columns=assets).mean()).fillna(DEFAULT_COST_BPS_PER_SIDE)
        C = cm.to_numpy() / 10_000
    else:
        if isinstance(cost_bps_per_side, dict):
            c = np.array([cost_bps_per_side.get(a, DEFAULT_COST_BPS_PER_SIDE) for a in assets]) / 10_000
        else:
            c = np.full(len(assets), float(cost_bps_per_side) / 10_000)
        C = np.broadcast_to(c, (len(target.index), len(assets)))

    E = target.to_numpy()
    R = np.nan_to_num(rets.to_numpy(), nan=0.0)
    RF = (np.zeros(len(target.index)) if cash_rate is None
          else cash_rate.reindex(target.index).fillna(0.0).to_numpy(dtype=float))
    n_days, n_assets = E.shape

    A = np.zeros(n_assets)   # posición en el activo
    K = np.zeros(n_assets)   # caja de cada "sleeve" (tasa 0)
    free_cash = 1.0          # capital todavía no asignado (solo antes del primer día)
    prev_active = None
    prev_e = np.full(n_assets, np.nan)
    prev_t = None

    out = np.full((n_days, len(PORTFOLIO_COLUMNS)), np.nan)
    trades = np.zeros((n_days, n_assets)) if return_trades else None
    for k in range(n_days):
        t = target.index[k]
        e = E[k]
        c = C[k]
        active = ~np.isnan(e)
        nav0 = A.sum() + K.sum() + free_cash

        rebal = _is_rebalance_day(rebalance, t, prev_t) or prev_active is None \
            or not np.array_equal(active, prev_active)
        V = A + K
        if rebal:
            V_new = np.where(active, nav0 / active.sum(), 0.0)
            A_new = np.where(active, _size_after_cost(np.nan_to_num(e), V_new, A, c), 0.0)
        else:
            V_new = V.copy()
            changed = active & (e != prev_e)
            A_new = np.where(changed, _size_after_cost(np.nan_to_num(e), V, A, c), A)
        if trades is not None:
            trades[k] = A_new - A
        trade = np.abs(A_new - A)
        cost_i = trade * c
        A = A_new
        K = V_new - A - cost_i
        free_cash = 0.0

        invested = A.sum()
        A = A * (1.0 + np.where(active, R[k], 0.0))
        K = K * (1.0 + RF[k])   # interés de la caja (tasa conocida en t-1)
        nav1 = A.sum() + K.sum()
        gross = (nav1 + cost_i.sum()) / nav0 - 1.0

        out[k] = [nav1 / nav0 - 1.0, gross, cost_i.sum() / nav0, trade.sum() / nav0,
                  invested / (nav0 - cost_i.sum()), active.sum(), np.nan]
        prev_active, prev_e, prev_t = active, e, t

    df = pd.DataFrame(out, index=target.index, columns=PORTFOLIO_COLUMNS)
    df["nav"] = (1 + df["net_return"]).cumprod()
    df["n_assets"] = df["n_assets"].astype(int)
    if return_trades:
        return df, pd.DataFrame(trades, index=target.index, columns=assets)
    return df


# --------------------------------------------------------------------------
# Exposiciones de los comparadores
# --------------------------------------------------------------------------

def active_mask_from_result(result: dict) -> pd.DataFrame:
    """Fechas en que cada activo está en la cartera de la estrategia (True/False)."""
    per_asset = result["backtest"]["per_asset"]
    idx = sorted(set().union(*[df.index for df in per_asset.values() if not df.empty]))
    mask = pd.DataFrame(False, index=pd.DatetimeIndex(idx), columns=list(per_asset))
    for t, df in per_asset.items():
        if not df.empty:
            mask.loc[df.index, t] = True
    return mask


def vol_target_exposure(log_returns: pd.Series, hysteresis_pp: float = HYSTERESIS_PP) -> pd.Series:
    """Exposición de vol targeting en fracción, point-in-time (usa hasta t-1)."""
    sigma = log_returns.rolling(VOL_TARGET_LOOKBACK).std().shift(1)
    sigma_star = sigma.expanding(min_periods=VOL_TARGET_MIN_HISTORY).median()
    raw = (sigma_star / sigma).clip(upper=1.0) * 100
    return apply_hysteresis(raw, band_pp=hysteresis_pp) / 100


def ma_filter_exposure(log_returns: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """1 si el precio de t-1 está sobre su media móvil de `window` ruedas, 0 si no."""
    price = np.exp(log_returns.fillna(0).cumsum())
    ma = price.rolling(window).mean()
    signal = (price > ma).astype(float).where(ma.notna())
    return signal.shift(1)


def build_comparator_exposures(result: dict) -> dict[str, pd.DataFrame]:
    """
    Matrices de exposición objetivo (fracción) de la estrategia y de cada
    comparador, todas sobre la misma máscara activo/fecha de la estrategia.
    """
    mask = active_mask_from_result(result)
    returns = result["returns"]
    strat = pd.DataFrame({t: result["exposures"][t] for t in mask.columns}).reindex(mask.index) / 100
    strat = strat.where(mask)

    def masked(fn):
        df = pd.DataFrame({t: fn(returns[t]) for t in mask.columns}).reindex(mask.index)
        return df.where(mask)

    ones = pd.DataFrame(1.0, index=mask.index, columns=mask.columns).where(mask)
    fixed = ones * strat.mean()

    return {
        "Estrategia": strat,
        "EW 100%": ones,
        "Exposición fija (media ex post)": fixed,
        "Vol targeting": masked(vol_target_exposure).fillna(1.0).where(mask),
        "MA200": masked(ma_filter_exposure).fillna(1.0).where(mask),
    }


# --------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------

def _simple_returns_frame(result: dict, columns) -> pd.DataFrame:
    return pd.DataFrame({t: log_to_simple(result["returns"][t]) for t in columns})


def run_comparators(
    result: dict,
    rebalance: str = DEFAULT_REBALANCE,
    merval_log_returns: pd.Series | None = None,
    cost_bps_per_side: dict | float | None = None,
    cash_rate: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Corre el motor para la estrategia y todos los comparadores."""
    cash_rate = cash_rate if cash_rate is not None else result.get("cash_rate")
    cost = cost_bps_per_side if cost_bps_per_side is not None else result["config"].cost_bps_per_side
    exposures = build_comparator_exposures(result)
    rets = _simple_returns_frame(result, exposures["Estrategia"].columns)
    out = {name: simulate_portfolio(rets, E, cost, rebalance, cash_rate=cash_rate) for name, E in exposures.items()}
    if merval_log_returns is not None:
        idx = out["Estrategia"].index
        m = log_to_simple(merval_log_returns).reindex(idx)
        out["Merval (índice)"] = pd.DataFrame({
            "net_return": m, "gross_return": m, "cost": 0.0, "turnover": 0.0,
            "exposure": 1.0, "n_assets": 1, "nav": (1 + m.fillna(0)).cumprod(),
        })
    return out


def comparison_table(portfolios: dict[str, pd.DataFrame], risk_free_rate=None) -> pd.DataFrame:
    """Métricas de desempeño + exposición media, turnover anual y costo total.
    Sharpe y Sortino en exceso de `risk_free_rate` (serie diaria o escalar anual)."""
    cols = []
    for name, df in portfolios.items():
        s = summary(df["net_return"].dropna(), name, risk_free_rate)
        s["Exposición media"] = df["exposure"].mean()
        s["Turnover anual"] = df["turnover"].sum() / (len(df) / 252)
        s["Costo total (pp)"] = 100 * df["cost"].sum()
        cols.append(s)
    return pd.concat(cols, axis=1)


def composition_periods(portfolio: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Tramos consecutivos con la misma cantidad de activos en la cartera."""
    n = portfolio["n_assets"]
    blocks = (n != n.shift()).cumsum()
    return [(g.index[0], g.index[-1], int(g.iloc[0])) for _, g in n.groupby(blocks)]


def metrics_by_period(portfolios: dict[str, pd.DataFrame], min_days: int = 60,
                      risk_free_rate=None) -> pd.DataFrame:
    """
    Retorno, volatilidad y max drawdown de cada cartera en cada tramo de
    composición de la estrategia. Tramos de menos de `min_days` ruedas se
    omiten (las métricas anualizadas no son informativas).
    """
    rows = []
    for start, end, n in composition_periods(portfolios["Estrategia"]):
        days = portfolios["Estrategia"].loc[start:end].shape[0]
        if days < min_days:
            continue
        for name, df in portfolios.items():
            r = df.loc[start:end, "net_return"].dropna()
            s = summary(r, name, risk_free_rate)
            rows.append({
                "tramo": f"{start.date()} → {end.date()}", "activos": n, "días": days,
                "cartera": name, "retorno_anual": s["Retorno anualizado"],
                "volatilidad": s["Volatilidad anualizada"], "max_drawdown": s["Max Drawdown"],
                "sharpe": s["Sharpe"],
            })
    return pd.DataFrame(rows)


def rebalance_sensitivity(result: dict, names=("Estrategia", "EW 100%")) -> pd.DataFrame:
    """Cuánto cambia el resultado según la frecuencia de rebalanceo supuesta."""
    exposures = build_comparator_exposures(result)
    rets = _simple_returns_frame(result, exposures["Estrategia"].columns)
    cost = result.get("cost_used", result["config"].cost_bps_per_side)
    rf = result.get("cash_rate")
    rows = []
    for rule in REBALANCE_RULES:
        for name in names:
            df = simulate_portfolio(rets, exposures[name], cost, rule, cash_rate=rf)
            s = summary(df["net_return"], name, rf)
            rows.append({"rebalanceo": rule, "cartera": name,
                         "retorno_anual": s["Retorno anualizado"], "sharpe": s["Sharpe"],
                         "max_drawdown": s["Max Drawdown"],
                         "costo_total_pp": 100 * df["cost"].sum()})
    
    bt = result["backtest"]
    e3 = {"Estrategia": bt["portfolio_net_return"], "EW 100%": result["benchmark_return"]}
    for name in names:
        if name in e3 and len(e3[name]):
            s = summary(e3[name], name, rf)
            rows.append({"rebalanceo": "E3: promedio diario sin costo", "cartera": name,
                         "retorno_anual": s["Retorno anualizado"], "sharpe": s["Sharpe"],
                         "max_drawdown": s["Max Drawdown"], "costo_total_pp": np.nan})
    return pd.DataFrame(rows)
