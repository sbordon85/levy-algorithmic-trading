"""
transaction_costs.py  (Entrega #4, punto 4: "Costos de transacción")

Incorporar o discutir comisiones, evaluar la liquidez de los activos operados
y, cuando corresponda, considerar slippage e impacto de mercado.

Qué ya existía (no se repite)
-----------------------------
  - backtest_engine.py: comisión de bróker 0,50 % + derecho de mercado 5 pb,
    ambos con IVA (66,55 pb por lado), más medio spread de Corwin-Schultz.
  - liquidity.py: costo variable diario, diagnóstico de participación en el
    ADV_20 (tope 10 %) y capacidad. Sus tablas están en res["liquidity"].
  - sensitivity.py: variantes de costo (solo fijo, medio, variable, p95, 350 pb).

Qué agrega este módulo
----------------------
 1. cost_scale_sweep      Costo por lado multiplicado por 0, 0,5, 1, 2, 3, 5:
                          ¿cuánto del resultado se va en costos y cuánto
                          aguantaría la estrategia?
 2. commission_scenarios  Comisión de bróker de 50 / 25 / 10 / 0 pb por lado
                          (escenarios, no tarifas de ningún bróker en
                          particular), manteniendo derecho de mercado, IVA y spread.
 3. cost_decomposition    De qué se compone el costo total: comisión + derecho
                          + IVA contra medio spread, por activo.
 4. market_impact_sensitivity
                          Impacto de mercado con el modelo de ley de potencias de
                          Mayer-Krebs (2025, cap. 4): C = b * sigma * (Q/V)^alpha,
                          con alpha = 0,40 y b = 0,35 (Deutsche Bank, 2018, mercado
                          de EE.UU.), para distintos tamaños de cartera (AUM).
 5. run_all               Corre todo y guarda CSV; incluye las tablas de liquidez
                          que ya trae res["liquidity"].

Sobre el impacto de mercado
---------------------------
  - El modelo de la fuente NO incluye el spread; en este trabajo el spread ya está
    en el costo por lado (medio spread de Corwin-Schultz = el término a * s / P con
    a = 0,5 de la extensión del libro), así que acá solo se agrega el término de
    impacto: b * sigma * (Q/V)^alpha.
  - Q = monto de la orden (fracción de la cartera operada x AUM, en pesos);
    V = ADV_20 en pesos conocido en t-1; sigma = desvío diario de los últimos 20
    retornos hasta t-1.
  - Los parámetros están calibrados para el mercado estadounidense. NO se
    calibraron para BYMA (se necesitarían operaciones ejecutadas): se reportan
    con b = 0,35 y con b multiplicado por 3 como escenario de mercado menos
    profundo. Es un análisis de sensibilidad, no una estimación.
  - Tope de participación: las órdenes que superan el 10 % del ADV se cuentan al
    impacto del 10 % (el resto se ejecutaría en ruedas siguientes, no modelado) y
    se informa qué porcentaje de órdenes supera el tope.
  - El AUM es una hipótesis: se barren varios valores. Conviene definirlo en
    DÓLARES constantes (pasar `fx`): un AUM fijo en pesos nominales se achica en
    términos reales a lo largo de una muestra con tanta inflación y devaluación,
    y la participación en el ADV de los primeros años queda enorme frente a la de
    los últimos (en pesos de cada fecha, el ADV crece muchísimo). Con `fx` el AUM
    de cada fecha es AUM_USD x USD/ARS oficial de esa fecha (cota: durante el cepo
    el oficial subestima la depreciación efectiva).

Uso (con `res` y `price_data` ya cargados, y robustness.py en la carpeta):

    import transaction_costs as tc
    out = tc.run_all(res, price_data, fx=fx_oficial)     # AUM en USD (recomendado)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import comparators as cmp
import liquidity as lq
from backtest_engine import DEFAULT_COST_BPS_PER_SIDE, FIXED_COST_BPS_PER_SIDE, IVA, MARKET_FEE_BPS
from robustness import EW, FIJA, STRAT, _ctx, _row, _targets

TRADING_DAYS = 252
IMPACT_ALPHA = 0.40
IMPACT_B = 0.35
SIGMA_WINDOW = 20


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _scale_cost(cost, m: float):
    if isinstance(cost, pd.DataFrame):
        return cost * m
    if isinstance(cost, dict):
        return {k: v * m for k, v in cost.items()}
    return float(cost) * m


def _shift_cost(cost, delta_bps: float):
    if isinstance(cost, pd.DataFrame):
        return cost + delta_bps
    if isinstance(cost, dict):
        return {k: v + delta_bps for k, v in cost.items()}
    return float(cost) + delta_bps


def _cost_frame(cost, index, columns) -> pd.DataFrame:
    """Costo por lado (pb) como DataFrame fechas x activos, completando faltantes como el motor."""
    if isinstance(cost, pd.DataFrame):
        cm = cost.reindex(index=index, columns=columns)
        return cm.fillna(cost.reindex(columns=columns).mean()).fillna(DEFAULT_COST_BPS_PER_SIDE)
    if isinstance(cost, dict):
        row = pd.Series({a: cost.get(a, DEFAULT_COST_BPS_PER_SIDE) for a in columns}, dtype=float)
    else:
        row = pd.Series(float(cost), index=columns, dtype=float)
    return pd.DataFrame(np.tile(row.to_numpy(), (len(index), 1)), index=index, columns=columns)


def _ports_with_cost(res: dict, cost, names=(STRAT, EW, FIJA)):
    targets, rets = _targets(res)
    _, reb, rf = _ctx(res)
    ports = {n: cmp.simulate_portfolio(rets, targets[n], cost, reb, cash_rate=rf) for n in names}
    return ports, rf


def _cost_row(ports: dict, rf) -> dict:
    row = _row(ports, rf)
    row["costo_total_pp"] = 100 * ports[STRAT]["cost"].sum()
    row["costo_total_pp_EW"] = 100 * ports[EW]["cost"].sum()
    return row


# --------------------------------------------------------------------------
# 1. Escala del costo
# --------------------------------------------------------------------------

def cost_scale_sweep(res: dict, mults=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0)) -> pd.DataFrame:
    """
    Multiplica el costo por lado de la corrida (fijo + medio spread, por activo y
    fecha) por cada factor. 0 = sin costos; 1 = base. Estrategia, EW y exposición
    fija pagan el mismo costo por unidad operada, pero operan distinto: EW solo
    rebalancea y la estrategia además cambia de exposición.
    """
    cost, _, _ = _ctx(res)
    rows = []
    for m in mults:
        ports, rf = _ports_with_cost(res, _scale_cost(cost, m))
        rows.append({"multiplo_costo": m, **_cost_row(ports, rf)})
    out = pd.DataFrame(rows).set_index("multiplo_costo")
    base = out.loc[1.0] if 1.0 in out.index else out.iloc[0]
    out["d_sharpe_vs_base"] = out["sharpe"] - base["sharpe"]
    out["d_retorno_vs_base"] = out["retorno_anual"] - base["retorno_anual"]
    return out


# --------------------------------------------------------------------------
# 2. Comisión de bróker
# --------------------------------------------------------------------------

def commission_scenarios(res: dict, broker_bps=(50.0, 25.0, 10.0, 0.0)) -> pd.DataFrame:
    """
    Reemplaza la comisión de bróker (50 pb por lado en la base, más IVA) por otros
    niveles, manteniendo derecho de mercado (5 pb), IVA y medio spread. Son
    escenarios para ver cuánto pesa la comisión; no son tarifas de ningún bróker.
    """
    cost, _, _ = _ctx(res)
    base_fixed = FIXED_COST_BPS_PER_SIDE
    rows = []
    for b in broker_bps:
        fixed = (b + MARKET_FEE_BPS) * (1 + IVA)
        ports, rf = _ports_with_cost(res, _shift_cost(cost, fixed - base_fixed))
        rows.append({"comision_broker_pb": b, "fijo_por_lado_pb": round(fixed, 2), **_cost_row(ports, rf)})
    out = pd.DataFrame(rows).set_index("comision_broker_pb")
    ref = out.loc[50.0] if 50.0 in out.index else out.iloc[0]
    out["d_sharpe_vs_base"] = out["sharpe"] - ref["sharpe"]
    return out


# --------------------------------------------------------------------------
# 3. Descomposición del costo
# --------------------------------------------------------------------------

def cost_decomposition(res: dict) -> pd.DataFrame:
    """
    Costo total de la estrategia por activo, separado en (i) comisión + derecho de
    mercado + IVA (fijo por lado) y (ii) medio spread. `chequeo_costo_total_pp`
    (en attrs) debe coincidir con el costo total del motor.
    """
    targets, rets = _targets(res)
    cost, reb, rf = _ctx(res)
    port, trades = cmp.simulate_portfolio(rets, targets[STRAT], cost, reb, return_trades=True, cash_rate=rf)
    nav_prev = port["nav"].shift(1).fillna(1.0)
    tr = trades.abs().div(nav_prev.reindex(trades.index), axis=0)       # fracción de la cartera operada
    C = _cost_frame(cost, tr.index, tr.columns)
    years = len(port) / TRADING_DAYS
    fixed = FIXED_COST_BPS_PER_SIDE
    rows = {}
    for a in tr.columns:
        cost_pp = 100 * (tr[a] * C[a]).sum() / 1e4
        fixed_pp = 100 * (tr[a] * np.minimum(fixed, C[a])).sum() / 1e4
        traded = tr[a] > 1e-9
        rows[a] = {
            "ordenes": int(traded.sum()),
            "turnover_anual": tr[a].sum() / years,
            "costo_por_lado_medio_pb": float((tr[a] * C[a]).sum() / tr[a].sum()) if tr[a].sum() > 0 else np.nan,
            "costo_total_pp": cost_pp,
            "comision_derecho_iva_pp": fixed_pp,
            "medio_spread_pp": cost_pp - fixed_pp,
        }
    out = pd.DataFrame(rows).T
    tot = out[["ordenes", "turnover_anual", "costo_total_pp", "comision_derecho_iva_pp", "medio_spread_pp"]].sum()
    tot["costo_por_lado_medio_pb"] = (tr * C).to_numpy().sum() / tr.to_numpy().sum() if tr.to_numpy().sum() else np.nan
    out.loc["total"] = tot
    out["pct_comision_derecho_iva"] = out["comision_derecho_iva_pp"] / out["costo_total_pp"]
    out["costo_anual_pp"] = out["costo_total_pp"] / years
    out.attrs["chequeo_costo_total_pp"] = (100 * port["cost"].sum(), float(out.loc["total", "costo_total_pp"]))
    return out


# --------------------------------------------------------------------------
# 4. Impacto de mercado (ley de potencias)
# --------------------------------------------------------------------------

def _fx_levels(fx: pd.Series, index) -> pd.Series:
    f = fx.dropna().sort_index()
    return f.reindex(f.index.union(index)).ffill().bfill().reindex(index)


def _aum_series(aum: float, fx, index) -> pd.Series:
    """AUM en pesos de cada fecha: constante (fx=None) o AUM en USD x tipo de cambio oficial."""
    if fx is None:
        return pd.Series(float(aum), index=index)
    return float(aum) * _fx_levels(fx, index)


def _impact_inputs(res: dict, price_data: dict, columns, index):
    sub = {t: price_data[t] for t in columns if t in price_data}
    adv = lq.adv_ars(sub).reindex(index=index, columns=columns)
    sigma = pd.DataFrame({t: res["returns"][t].rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW // 2).std().shift(1)
                          for t in columns}).reindex(index)
    return adv, sigma


def _sim_with_impact(res: dict, name: str, aum: float, b: float, alpha: float, cap: float,
                     price_data: dict, fx=None):
    targets, rets = _targets(res)
    cost, reb, rf = _ctx(res)
    cols = list(targets[name].columns)
    Cb = _cost_frame(cost, targets[name].index, cols)
    # primera pasada: operaciones con el costo base
    port0, trades = cmp.simulate_portfolio(rets, targets[name], Cb, reb, return_trades=True, cash_rate=rf)
    nav_prev = port0["nav"].shift(1).fillna(1.0)
    tr = trades.abs().div(nav_prev.reindex(trades.index), axis=0)
    adv, sigma = _impact_inputs(res, price_data, cols, tr.index)
    part = tr.mul(_aum_series(aum, fx, tr.index), axis=0).div(adv).where(tr > 1e-9)
    eff = part.clip(upper=cap)
    impact_bps = (b * sigma * eff ** alpha * 1e4).where(tr > 1e-9).fillna(0.0)
    # segunda pasada: costo base + impacto
    port = cmp.simulate_portfolio(rets, targets[name], Cb + impact_bps.reindex(Cb.index), reb, cash_rate=rf)
    traded = tr > 1e-9
    _ps = part.stack().dropna()  # pandas nuevo conserva NaN en stack(): sin dropna el % sobre el tope se diluye
    stats = {
        "ordenes": int(traded.to_numpy().sum()),
        "ordenes_sin_adv": int((traded & adv.isna()).to_numpy().sum()),
        "participacion_mediana": float(_ps.median()) if _ps.size else np.nan,
        "participacion_max": float(_ps.max()) if _ps.size else np.nan,
        "pct_ordenes_sobre_tope": float((_ps > cap).mean()) if _ps.size else np.nan,
        "impacto_medio_pb": float((tr * impact_bps).to_numpy().sum() / tr.to_numpy().sum()) if tr.to_numpy().sum() else np.nan,
    }
    return port, stats


def market_impact_sensitivity(res: dict, price_data: dict, aum_values=(1e8, 5e8, 2e9, 1e10),
                              b_values=(IMPACT_B, 3 * IMPACT_B), alpha: float = IMPACT_ALPHA,
                              cap: float = lq.ADV_CAP_PCT, fx: pd.Series | None = None) -> pd.DataFrame:
    """
    Estrategia, EW y exposición fija con costo base + impacto de mercado
    b * sigma * (Q/V)^alpha, para cada AUM (pesos) y cada coeficiente b. La
    primera fila es la base (sin impacto). Cada cartera calcula el impacto con
    sus propias operaciones. Columnas de liquidez (participación, % de órdenes
    sobre el tope) corresponden a la estrategia.

    `fx` (USD/ARS oficial, p. ej. `fx_oficial` del notebook): si se pasa, los
    valores de `aum_values` son DÓLARES y el AUM de cada fecha es aum x fx de esa
    fecha. Sin `fx`, son pesos nominales constantes (ver nota en el docstring del módulo).
    """
    cost, _, rf0 = _ctx(res)
    ports0, rf = _ports_with_cost(res, cost)
    rows = [{"aum": 0, "b": 0.0, **_cost_row(ports0, rf)}]
    for b in b_values:
        for aum in aum_values:
            ports, stats = {}, None
            for n in (STRAT, EW, FIJA):
                ports[n], st = _sim_with_impact(res, n, aum, b, alpha, cap, price_data, fx)
                if n == STRAT:
                    stats = st
            rows.append({"aum": aum, "b": b, **_cost_row(ports, rf), **stats})
    out = pd.DataFrame(rows).set_index(["aum", "b"])
    out.attrs["moneda_aum"] = "USD (x USD/ARS oficial de cada fecha)" if fx is not None else "ARS nominales constantes"
    base = out.iloc[0]
    out["d_sharpe_vs_base"] = out["sharpe"] - base["sharpe"]
    out["d_retorno_vs_base"] = out["retorno_anual"] - base["retorno_anual"]
    return out


# --------------------------------------------------------------------------
# 5. Capacidad en dólares por año
# --------------------------------------------------------------------------

def liquidity_capacity_usd(res: dict, price_data: dict, fx: pd.Series, cap: float = lq.ADV_CAP_PCT) -> dict:
    """
    ¿Son los activos suficientemente líquidos? Para cada orden de la estrategia,
    capacidad = valor de cartera máximo (USD de esa fecha, a USD/ARS oficial) con
    el que la orden no supera `cap` del ADV_20. Se resume por año y por activo con
    el percentil 5 y el mínimo: p. ej. "capacidad p5 = 2 millones" significa que con
    una cartera de 2 millones de dólares el 5 % de las órdenes superaría el tope.
    Es comparable entre años (los pesos de cada fecha no lo son).
    """
    targets, rets = _targets(res)
    cost, reb, rf = _ctx(res)
    port, trades = cmp.simulate_portfolio(rets, targets[STRAT], cost, reb, return_trades=True, cash_rate=rf)
    nav_prev = port["nav"].shift(1).fillna(1.0)
    tr = trades.abs().div(nav_prev.reindex(trades.index), axis=0)
    adv, _ = _impact_inputs(res, price_data, list(tr.columns), tr.index)
    fxl = _fx_levels(fx, tr.index)
    cap_usd = (cap * adv / tr).where(tr > 1e-9).div(fxl, axis=0)

    def _summ(df):
        v = df.stack().dropna()
        return {"ordenes": int(v.size), "capacidad_p5_USD": v.quantile(0.05) if v.size else np.nan,
                "capacidad_mediana_USD": v.median() if v.size else np.nan,
                "capacidad_min_USD": v.min() if v.size else np.nan}

    por_anio = pd.DataFrame({y: _summ(cap_usd[cap_usd.index.year == y]) for y in sorted(set(cap_usd.index.year))}).T
    por_activo = pd.DataFrame({a: _summ(cap_usd[[a]]) for a in cap_usd.columns}).T
    por_activo.loc["total"] = _summ(cap_usd)
    return {"por_anio": por_anio, "por_activo": por_activo}


# --------------------------------------------------------------------------
# 5b. Diagnóstico del impacto: ¿de dónde viene el impacto medio?
# --------------------------------------------------------------------------

def impact_diagnostic(res: dict, price_data: dict, fx: pd.Series, aum: float = 1e6,
                      b: float = IMPACT_B, alpha: float = IMPACT_ALPHA,
                      cap: float = lq.ADV_CAP_PCT, top: int = 10) -> dict:
    """
    Abre el "impacto medio (pb)" de market_impact_sensitivity, que es un promedio
    ponderado por el tamaño de la orden (sum(tr x impacto) / sum(tr)), de modo que
    unas pocas órdenes grandes (la compra inicial, el primer ingreso de cada activo)
    pueden dominarlo. Devuelve, para la estrategia y un AUM en USD:
      "ordenes":     una fila por orden (fecha, activo, fraccion_nav, participacion,
                     sigma, impacto_pb, tope = participación > cap, primera = primera
                     orden del activo)
      "resumen":     impacto medio ponderado y simple, todas las órdenes / sin primeras /
                     sin las que superan el tope / solo las que no superan el tope
      "por_activo":  nº de órdenes, % sobre el tope, participación mediana, impacto
                     simple mediano y ponderado, % del impacto total (ponderado)
      "por_anio":    lo mismo por año
      "top":         las `top` órdenes que más aportan al impacto total
    """
    targets, rets = _targets(res)
    cost, reb, rf = _ctx(res)
    cols = list(targets[STRAT].columns)
    Cb = _cost_frame(cost, targets[STRAT].index, cols)
    port0, trades = cmp.simulate_portfolio(rets, targets[STRAT], Cb, reb, return_trades=True, cash_rate=rf)
    nav_prev = port0["nav"].shift(1).fillna(1.0)
    tr = trades.abs().div(nav_prev.reindex(trades.index), axis=0)
    adv, sigma = _impact_inputs(res, price_data, cols, tr.index)
    part = tr.mul(_aum_series(aum, fx, tr.index), axis=0).div(adv).where(tr > 1e-9)
    imp = (b * sigma * part.clip(upper=cap) ** alpha * 1e4).where(tr > 1e-9)

    def _long(df, name):
        s = df.stack(); s.index.names = ["fecha", "activo"]; return s.rename(name)

    o = pd.concat([_long(tr, "fraccion_nav"), _long(part, "participacion"),
                   _long(sigma.reindex(tr.index), "sigma"), _long(imp, "impacto_pb")], axis=1)
    o = o[o["fraccion_nav"] > 1e-9].reset_index()
    o["tope"] = o["participacion"] > cap
    o["primera"] = ~o.duplicated("activo", keep="first")
    o["anio"] = pd.to_datetime(o["fecha"]).dt.year
    o["aporte"] = o["fraccion_nav"] * o["impacto_pb"]

    def _w(d):
        d = d.dropna(subset=["impacto_pb"])
        w = d["fraccion_nav"].sum()
        return {"ordenes": len(d), "impacto_ponderado_pb": (d["aporte"].sum() / w) if w else np.nan,
                "impacto_simple_medio_pb": d["impacto_pb"].mean(), "impacto_simple_mediano_pb": d["impacto_pb"].median()}

    resumen = pd.DataFrame({"todas": _w(o), "sin_primeras": _w(o[~o["primera"]]),
                            "sin_sobre_tope": _w(o[~o["tope"]]),
                            "solo_sobre_tope": _w(o[o["tope"]]),
                            "sin_primeras_ni_tope": _w(o[~o["primera"] & ~o["tope"]])}).T

    def _grp(key):
        g = o.groupby(key)
        d = pd.DataFrame({"ordenes": g.size(), "pct_sobre_tope": g["tope"].mean(),
                          "participacion_mediana": g["participacion"].median(),
                          "impacto_simple_mediano_pb": g["impacto_pb"].median()})
        d["impacto_ponderado_pb"] = g.apply(lambda x: x["aporte"].sum() / x["fraccion_nav"].sum())
        d["pct_del_impacto_total"] = g["aporte"].sum() / o["aporte"].sum()
        return d

    topo = o.sort_values("aporte", ascending=False).head(top)[
        ["fecha", "activo", "fraccion_nav", "participacion", "impacto_pb", "tope", "primera"]]
    out = {"ordenes": o, "resumen": resumen, "por_activo": _grp("activo"), "por_anio": _grp("anio"), "top": topo}
    out["resumen"].attrs["aum_usd"] = aum
    return out


# --------------------------------------------------------------------------
# Todo junto
# --------------------------------------------------------------------------

def run_all(res: dict, price_data: dict | None = None, prefix: str = "costos_", save: bool = True,
            aum_values=None, fx: pd.Series | None = None, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    out: dict = {}

    def step(name, fn):
        t0 = time.time()
        out[name] = fn()
        log(f"  {name}: {time.time() - t0:.0f} s")

    log("Costos de transacción (punto 4)")
    step("descomposicion", lambda: cost_decomposition(res))
    chk = out["descomposicion"].attrs["chequeo_costo_total_pp"]
    log(f"  chequeo costo total (motor vs. descomposición): {chk[0]:.4f} vs {chk[1]:.4f} pp")
    step("escala_costos", lambda: cost_scale_sweep(res))
    step("comision", lambda: commission_scenarios(res))
    if price_data:
        aums = aum_values or ((2.5e5, 1e6, 5e6, 2e7) if fx is not None else (1e8, 5e8, 2e9, 1e10))
        step("impacto_mercado", lambda: market_impact_sensitivity(res, price_data, aums, fx=fx))
        if fx is not None:
            cu = liquidity_capacity_usd(res, price_data, fx)
            out["capacidad_usd_por_anio"], out["capacidad_usd_por_activo"] = cu["por_anio"], cu["por_activo"]
    liq = res.get("liquidity")
    if liq:
        out["liquidez_por_activo"] = liq["por_activo"]
        out["liquidez_por_anio"] = liq["por_anio"]
    if save:
        for name, df in out.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(f"{prefix}{name}.csv")
    return out
