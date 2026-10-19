"""
execution_prices.py  (Entrega #4, punto 3: "Supuestos de ejecución")

Pregunta que responde: ¿cambia el resultado según A QUÉ PRECIO se ejecuta la
decisión de la estrategia? Compara la convención base (cierre de t-1) contra
el open de t, aproximaciones al VWAP y el cierre de t.

Por qué no hay VWAP
-------------------
El VWAP = sum(P_i * V_i) / sum(V_i) necesita datos intradiarios (operaciones,
minutos u horas). yfinance entrega la serie diaria completa de la muestra
(OHLCV), pero lo más fino que ofrece es 1 hora y con una historia de unos
1.085 días, que no cubre toda la muestra ni los episodios del catálogo (ver
`data_loader.py`). Se usan entonces aproximaciones diarias:

  - "open"    : open de t (supuesto estándar de los backtests; el libro lo
                usa para las órdenes de mercado: ejecución en el open de la
                barra siguiente).
  - "ohlc4"   : (open + high + low + close) / 4 de t.
  - "tipico"  : precio típico (high + low + close) / 3 de t. Es la aproximación
                práctica al VWAP que propone Mayer-Krebs (2025, cap. 4).
  - "cierre"  : cierre de t (el otro extremo del día).
  - base      : cierre de t-1 (lo que supone el motor desde la Entrega #3).

Ninguna usa el volumen, así que NO son VWAP: son aproximaciones y deben
declararse como tales.

Cómo se modela
--------------
El retorno de cierre a cierre de t se parte en dos tramos alrededor del precio
de ejecución P:

    tramo previo  = P / cierre_{t-1} - 1     lo gana la posición que ya se tenía
    tramo posterior = cierre_t / P - 1       lo gana la posición DESPUÉS de operar

con (1 + previo)(1 + posterior) = 1 + retorno de t, que es el mismo retorno
simple que usa el motor (incluye dividendos, porque el tramo previo se calcula
como el resto). Con ejecución al cierre de t-1 el tramo previo es 0 y el
posterior es el retorno completo: reproduce `comparators.simulate_portfolio`
exactamente (se verifica en `check_base`). Cuando falta el precio de un día
(o el tramo es absurdo) esa celda usa la convención base y se informa cuántas
fueron.

`simulate_portfolio_exec` copia la lógica de `comparators.simulate_portfolio`
(tenencias, bolsillos, costo por lado, dimensionamiento después de costos,
caja con tasa) y agrega solo el tramo previo y un costo extra opcional.

Uso (desde el notebook, con `res` y `price_data` ya cargados):

    import execution_prices as ep
    out = ep.run_all(res, price_data)
    out["ejecucion"], out["overnight_intradia"]
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import comparators as cmp
from comparators import _simple_returns_frame, _size_after_cost
from robustness import EW, FIJA, STRAT, _ctx, _first_full_date, _row, _targets
from backtest_engine import DEFAULT_COST_BPS_PER_SIDE

TRADING_DAYS = 252
MAX_LEG = 0.40          # un tramo mayor a ±40 % se considera dato dudoso y vuelve a la base

PRICE_KINDS = {
    "open": "open de t",
    "ohlc4": "promedio OHLC de t",
    "tipico": "precio típico (H+L+C)/3 de t",
    "cierre": "cierre de t",
}


# --------------------------------------------------------------------------
# Precios de ejecución -> tramos previo y posterior
# --------------------------------------------------------------------------

def _exec_price(df: pd.DataFrame, kind: str) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    if kind == "open":
        return o
    if kind == "ohlc4":
        return (o + h + l + c) / 4.0
    if kind == "tipico":
        return (h + l + c) / 3.0
    if kind == "cierre":
        return c
    raise ValueError(f"kind desconocido: {kind!r}. Opciones: {tuple(PRICE_KINDS)}")


def exec_returns(res: dict, price_data: dict, kind: str | None, columns=None):
    """
    Devuelve (previo, posterior, n_fallback): dos DataFrames fechas x activos con
    los tramos del retorno simple de t alrededor del precio de ejecución, y un
    dict con la cantidad de celdas que volvieron a la convención base.
    `kind=None` es la base (cierre de t-1): previo = 0, posterior = retorno.
    """
    cols = list(columns) if columns is not None else list(res["returns"])
    r = _simple_returns_frame(res, cols)
    if kind is None:
        return r * 0.0, r, {c: 0 for c in cols}
    pre = pd.DataFrame(0.0, index=r.index, columns=cols)
    post = r.copy()
    fallback = {}
    for t in cols:
        df = price_data[t].reindex(r.index)
        p = _exec_price(df, kind)
        c = df["Close"]
        po = c / p - 1.0
        pr = (1.0 + r[t]) / (1.0 + po) - 1.0
        bad = (p.isna() | c.isna() | (p <= 0) | (c <= 0) | po.isna() | pr.isna()
               | (po.abs() > MAX_LEG) | (pr.abs() > MAX_LEG)) & r[t].notna()
        pre[t] = pr.where(~bad, 0.0)
        post[t] = po.where(~bad, r[t])
        fallback[t] = int(bad.sum())
    return pre, post, fallback


# --------------------------------------------------------------------------
# Motor con tramo previo
# --------------------------------------------------------------------------

def simulate_portfolio_exec(pre_returns: pd.DataFrame, post_returns: pd.DataFrame,
                            target_exposure: pd.DataFrame, cost_bps_per_side=None,
                            rebalance: str = cmp.DEFAULT_REBALANCE, cash_rate: pd.Series | None = None,
                            extra_cost_bps: float = 0.0) -> pd.DataFrame:
    """
    Igual que `comparators.simulate_portfolio`, pero el retorno diario se aplica
    en dos tramos: primero el previo (a la posición que se tenía), después se
    opera al precio de ejecución, y recién ahí el posterior (a la posición nueva).
    `extra_cost_bps` suma un costo adicional por lado (prueba de estrés).
    """
    if cost_bps_per_side is None:
        cost_bps_per_side = cmp.COST_BPS_PER_SIDE_BY_ASSET
    if rebalance not in cmp.REBALANCE_RULES:
        raise ValueError(f"Regla de rebalanceo desconocida: {rebalance!r}. Opciones: {cmp.REBALANCE_RULES}")
    target = target_exposure.astype(float)
    target = target[target.notna().any(axis=1)]
    if target.empty:
        return pd.DataFrame(columns=cmp.PORTFOLIO_COLUMNS)
    assets = list(target.columns)
    idx = target.index
    R1 = np.nan_to_num(pre_returns.reindex(index=idx, columns=assets).astype(float).to_numpy(), nan=0.0)
    R2 = np.nan_to_num(post_returns.reindex(index=idx, columns=assets).astype(float).to_numpy(), nan=0.0)

    if isinstance(cost_bps_per_side, pd.DataFrame):
        cm = cost_bps_per_side.reindex(index=idx, columns=assets)
        cm = cm.fillna(cost_bps_per_side.reindex(columns=assets).mean()).fillna(DEFAULT_COST_BPS_PER_SIDE)
        C = cm.to_numpy() / 10_000
    else:
        if isinstance(cost_bps_per_side, dict):
            c0 = np.array([cost_bps_per_side.get(a, DEFAULT_COST_BPS_PER_SIDE) for a in assets]) / 10_000
        else:
            c0 = np.full(len(assets), float(cost_bps_per_side) / 10_000)
        C = np.broadcast_to(c0, (len(idx), len(assets)))
    C = C + extra_cost_bps / 10_000

    E = target.to_numpy()
    RF = (np.zeros(len(idx)) if cash_rate is None
          else cash_rate.reindex(idx).fillna(0.0).to_numpy(dtype=float))
    n_days, n_assets = E.shape

    A = np.zeros(n_assets)
    K = np.zeros(n_assets)
    free_cash = 1.0
    prev_active = None
    prev_e = np.full(n_assets, np.nan)
    prev_t = None
    out = np.full((n_days, len(cmp.PORTFOLIO_COLUMNS)), np.nan)

    for k in range(n_days):
        t = idx[k]
        e = E[k]
        c = C[k]
        active = ~np.isnan(e)
        nav0 = A.sum() + K.sum() + free_cash

        A = A * (1.0 + R1[k])                       # tramo previo: la posición que se tenía
        nav_mid = A.sum() + K.sum() + free_cash
        rebal = cmp._is_rebalance_day(rebalance, t, prev_t) or prev_active is None \
            or not np.array_equal(active, prev_active)
        V = A + K
        if rebal:
            V_new = np.where(active, nav_mid / active.sum(), 0.0)
            A_new = np.where(active, _size_after_cost(np.nan_to_num(e), V_new, A, c), 0.0)
        else:
            V_new = V.copy()
            changed = active & (e != prev_e)
            A_new = np.where(changed, _size_after_cost(np.nan_to_num(e), V, A, c), A)
        trade = np.abs(A_new - A)
        cost_i = trade * c
        A = A_new
        K = V_new - A - cost_i
        free_cash = 0.0

        invested = A.sum()
        A = A * (1.0 + np.where(active, R2[k], 0.0))   # tramo posterior: la posición nueva
        K = K * (1.0 + RF[k])
        nav1 = A.sum() + K.sum()
        gross = (nav1 + cost_i.sum()) / nav0 - 1.0

        out[k] = [nav1 / nav0 - 1.0, gross, cost_i.sum() / nav0, trade.sum() / nav0,
                  invested / (nav_mid - cost_i.sum()), active.sum(), np.nan]
        prev_active, prev_e, prev_t = active, e, t

    df = pd.DataFrame(out, index=idx, columns=cmp.PORTFOLIO_COLUMNS)
    df["nav"] = (1 + df["net_return"]).cumprod()
    df["n_assets"] = df["n_assets"].astype(int)
    return df


# --------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------

def _ports(res: dict, price_data: dict, kind: str | None, extra_cost_bps: float = 0.0, desde=None):
    targets, _ = _targets(res)
    if desde is not None:
        d = _first_full_date(res) if desde == "primera_cartera_completa" else pd.Timestamp(desde)
        strat = targets[STRAT].loc[d:]
        ones = targets[EW].loc[d:]
        targets = {STRAT: strat, EW: ones, FIJA: ones * strat.mean()}   # la exposición fija usa la media del tramo
    cols = list(targets[STRAT].columns)
    pre, post, fb = exec_returns(res, price_data, kind, cols)
    cost, reb, rf = _ctx(res)
    ports = {n: simulate_portfolio_exec(pre, post, targets[n], cost, reb, rf, extra_cost_bps)
             for n in (STRAT, EW, FIJA)}
    return ports, fb, rf


def check_base(res: dict, price_data: dict | None = None, tol: float = 1e-10) -> float:
    """Diferencia máxima entre la base de este módulo y comparators.simulate_portfolio (debe ser ~0)."""
    ports, _, _ = _ports(res, price_data or {}, None)
    ref = res["portfolios"]["Estrategia"]["net_return"]
    d = (ports[STRAT]["net_return"] - ref.reindex(ports[STRAT].index)).abs().max()
    if d > tol:
        print(f"[WARN] la base difiere de comparators.simulate_portfolio en {d:.2e}")
    return float(d)


def execution_comparison(res: dict, price_data: dict,
                         variants=("open", "ohlc4", "tipico", "cierre"),
                         stress_bps=(25.0, 50.0), desde=None) -> pd.DataFrame:
    """
    Estrategia, EW y exposición fija con cada precio de ejecución. Las filas
    `open + X pb` suman un costo extra por lado a la ejecución en el open (la
    apertura suele tener spread más amplio que el resto de la rueda).
    Columnas: niveles y diferencias contra EW y contra la exposición fija (las
    mismas que en robustness.py), más el costo total y la variación contra la base.

    `desde`: fecha ("2022-12-26") o "primera_cartera_completa" para evaluar solo
    desde que están los cinco activos. Sirve para separar el efecto del precio
    de ejecución del efecto de las COMPRAS INICIALES: con ejecución posterior
    al cierre de t-1, cada activo que entra a la cartera pierde el retorno de
    su primer día, y con un NAV que se multiplica cientos de veces en pesos,
    un solo día grande al inicio mueve el retorno anualizado.
    """
    plan = [("cierre de t-1 (base)", None, 0.0)]
    plan += [(PRICE_KINDS[v], v, 0.0) for v in variants]
    plan += [(f"open de t + {int(s)} pb por lado", "open", float(s)) for s in stress_bps]
    rows, fb_all = [], {}
    for label, kind, extra in plan:
        ports, fb, rf = _ports(res, price_data, kind, extra, desde)
        fb_all[label] = fb
        row = _row(ports, rf)
        row["costo_total_pp"] = 100 * ports[STRAT]["cost"].sum()
        row["celdas_con_base"] = int(sum(fb.values()))
        rows.append({"ejecucion": label, **row})
    out = pd.DataFrame(rows).set_index("ejecucion")
    b = out.iloc[0]
    out["d_sharpe_vs_base"] = out["sharpe"] - b["sharpe"]
    out["d_mdd_vs_base"] = out["max_drawdown"] - b["max_drawdown"]
    out["d_retorno_vs_base"] = out["retorno_anual"] - b["retorno_anual"]
    out.attrs["celdas_con_base_por_activo"] = pd.DataFrame(fb_all).T
    return out


def overnight_intraday(res: dict, price_data: dict) -> pd.DataFrame:
    """
    Por activo: de dónde viene el retorno diario, de la noche (cierre de t-1 al
    open de t) o de la rueda (open a cierre de t). Sirve para entender por qué
    importa el precio de ejecución: si el retorno se concentra en la noche,
    ejecutar al open o al cierre previo cambia mucho lo que se captura.
    """
    pre, post, fb = exec_returns(res, price_data, "open")
    rows = {}
    for t in pre.columns:
        a, b = pre[t].dropna(), post[t].dropna()
        rows[t] = {
            "dias": len(a),
            "noche_media_diaria_pb": 1e4 * a.mean(),
            "rueda_media_diaria_pb": 1e4 * b.mean(),
            "noche_vol_diaria_pct": 100 * a.std(),
            "rueda_vol_diaria_pct": 100 * b.std(),
            "dias_con_gap_mayor_2pct": float((a.abs() > 0.02).mean()),
            "celdas_dudosas": fb[t],
        }
    return pd.DataFrame(rows).T


def run_all(res: dict, price_data: dict, prefix: str = "sens_ejec_", save: bool = True,
            verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    out: dict = {}
    t0 = time.time()
    d = check_base(res, price_data)
    log(f"  verificación de la base: dif. máxima {d:.1e}")
    out["ejecucion"] = execution_comparison(res, price_data)
    log(f"  ejecucion: {time.time() - t0:.0f} s")
    out["ejecucion_5_activos"] = execution_comparison(res, price_data, desde="primera_cartera_completa")
    log(f"  ejecucion_5_activos: {time.time() - t0:.0f} s")
    out["overnight_intradia"] = overnight_intraday(res, price_data)
    if save:
        for name, df in out.items():
            df.to_csv(f"{prefix}{name}.csv")
    return out
