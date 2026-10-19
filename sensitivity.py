"""
sensitivity.py  (Entrega #4, Fase 6)

Harness de sensibilidad de `harness_sensibilidad_parametros.qmd`, conectado
al pipeline real. Responde la aclaración general del profesor: probar cuánto
dependen los resultados de los supuestos.

Qué se mide en cada variante
----------------------------
No alcanza con el Sharpe de la estrategia: con este universo, cualquier
variante que esté ~90% invertida tiene un Sharpe parecido al del mercado. Por
eso cada fila reporta, además, la diferencia contra:
  - EW 100% (¿protege?): dif_mdd_vs_EW > 0 = cae menos que el benchmark;
  - exposición fija a la media de ESA variante (¿aporta el timing?):
    dif_sharpe_vs_fija > 0 = el momento en que se reduce la exposición suma
    algo más que el nivel medio de exposición.
La conclusión de la Fase 4 ("protege por nivel, no por timing") es robusta
si dif_sharpe_vs_fija se mantiene cerca de 0 en todas las variantes.

Dos tipos de parámetro
----------------------
- De SEÑAL (reajustan el logit): u (umbral de salto, u = k*sigma), h
  (horizonte del evento), criterio de W. Cada variante corre el pipeline
  (con W fijos y NIG desde caché; u y h se derivan de los mismos parámetros).
- De CARTERA (no tocan la señal): banda de histéresis (se reaplica sobre la
  exposición cruda del panel), costo, frecuencia de rebalanceo y piso por
  drawdown. Son baratos: solo se re-simula la cartera.

Todas las variantes (incluida la base) usan el mismo `refit_step`, para que
las diferencias no vengan del reajuste del logit.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd

import macro_data as md
from comparators import _simple_returns_frame, build_comparator_exposures, simulate_portfolio
from liquidity import COST_MODES, apply_drawdown_floor, cost_for_mode
from performance_metrics import summary
from protection import bootstrap_differences
from signal_generator import apply_hysteresis
from window_selection import EPISODE_CATALOG

SIGNAL_PARAMS = {
    "u_multiplier": [1.5, 1.75, 2.0, 2.5, 3.0],
    "horizon": [3, 5, 7, 10],
}
PORTFOLIO_PARAMS = {
    "hysteresis_pp": [0.0, 2.0, 5.0, 8.0, 12.0],
    "costo": ["solo fijo", "medio", "variable", "p95", "350 pb"],
    "rebalance": ["diario", "mensual", "trimestral", "nunca"],
}
DRAWDOWN_GRID = {"d_max": [0.10, 0.20, 0.30], "kappa": [1.0, 2.0, 4.0], "e_min": [25.0, 50.0]}
W_ALTERNATIVES = [("estimadores", "suma"), ("estandarizado", "max"), ("sin_mu", "max")]
# Opción B (Entrega #4): W corta y FIJA, la misma para todos los activos.
FIXED_WINDOWS = (500, 750, 1000)


# --------------------------------------------------------------------------
# Evaluación de una variante
# --------------------------------------------------------------------------

def _metrics_row(strategy: pd.DataFrame, ew: pd.DataFrame, fixed: pd.DataFrame, rf=None) -> dict:
    s, e, f = (summary(x["net_return"], risk_free_rate=rf) for x in (strategy, ew, fixed))
    return {
        "sharpe": s["Sharpe"], "retorno_anual": s["Retorno anualizado"], "max_drawdown": s["Max Drawdown"],
        "calmar": s["Calmar"], "exposicion_media": strategy["exposure"].mean(),
        "turnover_anual": strategy["turnover"].sum() / (len(strategy) / 252),
        "costo_total_pp": 100 * strategy["cost"].sum(),
        "dif_sharpe_vs_EW": s["Sharpe"] - e["Sharpe"],
        "dif_mdd_vs_EW": s["Max Drawdown"] - e["Max Drawdown"],
        "dif_sharpe_vs_fija": s["Sharpe"] - f["Sharpe"],
        "dif_mdd_vs_fija": s["Max Drawdown"] - f["Max Drawdown"],
        "dias": len(strategy),
    }


def evaluate_portfolio_variant(result: dict, exposures_pct: dict | None = None, cost=None,
                               rebalance: str | None = None) -> dict:
    """Re-simula estrategia, EW y exposición fija con otra exposición/costo/rebalanceo."""
    res = dict(result)
    if exposures_pct is not None:
        res["exposures"] = exposures_pct
    E = build_comparator_exposures(res)
    rets = _simple_returns_frame(res, E["Estrategia"].columns)
    cost = result.get("cost_used", result["config"].cost_bps_per_side) if cost is None else cost
    rebalance = rebalance or result["config"].rebalance
    rf = result.get("cash_rate")
    ports = {k: simulate_portfolio(rets, E[k], cost, rebalance, cash_rate=rf)
             for k in ("Estrategia", "EW 100%", "Exposición fija (media ex post)")}
    return _metrics_row(ports["Estrategia"], ports["EW 100%"], ports["Exposición fija (media ex post)"], rf)


def rehysteresis(result: dict, band_pp: float) -> dict[str, pd.Series]:
    """Reaplica la banda de histéresis a la exposición cruda del panel (antes de la banda)."""
    panel = result["panel"].dropna(subset=["exposure_raw"])
    out = {}
    for t, e in result["exposures_signal"].items():
        raw = panel[panel["asset"] == t].set_index("date")["exposure_raw"].reindex(e.index)
        out[t] = apply_hysteresis(raw, band_pp=band_pp) if band_pp > 0 else raw.copy()
    return out


def _cost_variant(label: str, result: dict, price_data):
    if label == "solo fijo":
        from backtest_engine import FIXED_COST_BPS_PER_SIDE
        return FIXED_COST_BPS_PER_SIDE
    if label == "350 pb":
        return 350.0
    return cost_for_mode(label, price_data, result["config"].spread_window)[0]


# --------------------------------------------------------------------------
# OFAT
# --------------------------------------------------------------------------

def run_ofat(result: dict, returns_by_asset: dict, price_data: dict | None = None,
             signal_params: dict | None = None, portfolio_params: dict | None = None,
             w_alternatives=W_ALTERNATIVES, windows_table: pd.DataFrame | None = None,
             refit_step: int = 5, verbose: bool = True) -> pd.DataFrame:
    """
    Varía un parámetro por vez alrededor de la configuración de `result`.
    La fila 'base' se recalcula con el mismo refit_step que las variantes.
    `windows_table` (window_criteria_comparison.csv) habilita las variantes
    de criterio de W.
    """
    from main import PipelineConfig, run_pipeline

    signal_params = SIGNAL_PARAMS if signal_params is None else signal_params
    portfolio_params = PORTFOLIO_PARAMS if portfolio_params is None else portfolio_params
    base_cfg = result["config"]
    base_kwargs = {**asdict(base_cfg), "windows": dict(result["windows"]), "refit_step": refit_step}
    log = print if verbose else (lambda *a, **k: None)

    def run(**over):
        t0 = time.time()
        cfg = PipelineConfig(**{**base_kwargs, **over})
        r = run_pipeline(returns_by_asset, cfg, verbose=False, price_data=price_data,
                         cash_rate=result.get("cash_rate"))
        log(f"  {over}: {time.time() - t0:.0f} s")
        return r

    rows = []
    base = run()
    rows.append({"parametro": "base", "valor": "base", **evaluate_portfolio_variant(base)})

    for p, values in signal_params.items():
        for v in values:
            if v == getattr(base_cfg, p):
                continue
            r = run(**{p: v})
            if r["portfolios"]:
                rows.append({"parametro": p, "valor": v, **evaluate_portfolio_variant(r)})

    if windows_table is not None:
        from protection import windows_from_criteria_table
        wins = windows_from_criteria_table(windows_table)
        for crit, comb in w_alternatives:
            if (crit, comb) in wins and (crit, comb) != (base_cfg.window_criterion, base_cfg.window_combine):
                r = run(windows=wins[(crit, comb)], window_criterion=crit, window_combine=comb)
                if r["portfolios"]:
                    rows.append({"parametro": "criterio_W", "valor": f"{crit}/{comb}",
                                 **evaluate_portfolio_variant(r)})

    for p, values in portfolio_params.items():
        for v in values:
            if p == "hysteresis_pp":
                if v == base_cfg.hysteresis_pp:
                    continue
                row = evaluate_portfolio_variant(base, exposures_pct=rehysteresis(base, v))
            elif p == "costo":
                if v == base_cfg.cost_mode:
                    continue
                row = evaluate_portfolio_variant(base, cost=_cost_variant(v, base, price_data))
            elif p == "rebalance":
                if v == base_cfg.rebalance:
                    continue
                row = evaluate_portfolio_variant(base, rebalance=v)
            else:
                raise ValueError(f"Parámetro de cartera desconocido: {p}")
            rows.append({"parametro": p, "valor": v, **row})
    out = pd.DataFrame(rows)
    out.attrs["base"] = base
    return out


def tornado_table(ofat: pd.DataFrame, metric: str = "dif_sharpe_vs_fija") -> pd.DataFrame:
    """Rango (máx − mín) de la métrica por parámetro, incluyendo la base."""
    base_val = ofat.loc[ofat["parametro"] == "base", metric].iloc[0]
    rows = []
    for p, g in ofat[ofat["parametro"] != "base"].groupby("parametro"):
        vals = pd.concat([g[metric], pd.Series([base_val])])
        rows.append({"parametro": p, "min": vals.min(), "max": vals.max(), "rango": vals.max() - vals.min(),
                     "valor_min": g.loc[g[metric].idxmin(), "valor"], "valor_max": g.loc[g[metric].idxmax(), "valor"]})
    t = pd.DataFrame(rows).sort_values("rango", ascending=False).reset_index(drop=True)
    t.attrs["base"] = base_val
    return t


# --------------------------------------------------------------------------
# Robustez con W corta y fija (opción B, feedback Entrega #3: "más historia")
# --------------------------------------------------------------------------

def _episodes_covered(index: pd.DatetimeIndex) -> list[str]:
    """Episodios del catálogo con al menos una fecha dentro del período evaluado."""
    lo, hi = index.min(), index.max()
    return [n for n, ds in EPISODE_CATALOG.items() if any(lo <= pd.Timestamp(d) <= hi for d in ds)]


def _first_full_date(portfolio: pd.DataFrame) -> pd.Timestamp:
    """Primera fecha con TODOS los activos en la cartera."""
    n = portfolio["n_assets"]
    return n[n == n.max()].index[0]


_PORT_KEYS = {"s": "Estrategia", "ew": "EW 100%", "fija": "Exposición fija (media ex post)"}


def _slice_metrics(ports: dict, start: pd.Timestamp, rf=None) -> dict:
    return {tag: summary(ports[name]["net_return"].loc[start:], risk_free_rate=rf)
            for tag, name in _PORT_KEYS.items()}


def run_fixed_window_grid(result: dict, returns_by_asset: dict, price_data: dict | None = None,
                          windows=FIXED_WINDOWS, refit_step: int = 5, reps: int = 1000,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Robustez con ventana corta FIJA (la misma W para los 5 activos).

    Responde al feedback de la Entrega #3 ("incorporar más historia utilizable
    para evaluar varios años"). NO cambia la especificación principal
    (criterio 'estimadores'/'max'): es una prueba de si las conclusiones
    (protege por nivel de exposición, no por timing) se sostienen con más
    historia. Advertencia que hay que declarar: estas W NO pasan el criterio de
    estabilización de W_E, así que el NIG es más ruidoso.

    No son candidatas a elegir: no deben entrar a `run_configuration_grid`
    ni al PBO. Si se las cuenta en el Deflated Sharpe, sumarlas a N.

    Cada fila reporta (una por W, más la base para comparar):
      - cobertura: primera fecha, primera fecha con los 5 activos, años
        evaluables y episodios del catálogo cubiertos;
      - métricas en el período propio de cada variante;
      - las mismas métricas en el período COMÚN a todas las variantes
        (prefijo `com_`), que es la comparación de igual a igual: el Sharpe de
        períodos distintos no es comparable;
      - bootstrap estacionario pareado (IC 95%) de Sharpe(estrategia) −
        Sharpe(exposición fija) y de MDD(estrategia) − MDD(EW), en el período
        propio, y un veredicto sobre el timing.
    Todas las variantes usan el mismo `refit_step` (como run_ofat).
    """
    from main import PipelineConfig, run_pipeline

    base_cfg = result["config"]
    base_kwargs = {**asdict(base_cfg), "refit_step": refit_step}
    log = print if verbose else (lambda *a, **k: None)

    variants = [(f"base ({base_cfg.window_criterion}/{base_cfg.window_combine})",
                 {**base_kwargs, "windows": dict(result["windows"])})]
    for w in windows:
        variants.append((f"W={int(w)}", {**base_kwargs, "windows": {t: int(w) for t in returns_by_asset}}))

    runs = {}
    for label, kw in variants:
        t0 = time.time()
        r = run_pipeline(returns_by_asset, PipelineConfig(**kw), verbose=False, price_data=price_data,
                         cash_rate=result.get("cash_rate"))
        log(f"  {label}: {time.time() - t0:.0f} s")
        runs[label] = r

    valid = {k: r for k, r in runs.items() if r["portfolios"]}
    if not valid:
        return pd.DataFrame()
    common_start = max(r["portfolios"]["Estrategia"].index[0] for r in valid.values())

    rows = []
    for label, r in runs.items():
        if label not in valid:
            rows.append({"variante": label, "nota": "sin exposición válida"})
            continue
        ports = r["portfolios"]
        strat = ports["Estrategia"]
        full_start = _first_full_date(strat)
        rf = result.get("cash_rate")
        own = _slice_metrics(ports, strat.index[0], rf)
        com = _slice_metrics(ports, common_start, rf)
        b_fija = bootstrap_differences(strat["net_return"], ports[_PORT_KEYS["fija"]]["net_return"], reps,
                                       risk_free=rf)
        b_ew = bootstrap_differences(strat["net_return"], ports[_PORT_KEYS["ew"]]["net_return"], reps,
                                     risk_free=rf)
        sh = b_fija.set_index("métrica").loc["sharpe"]
        md = b_ew.set_index("métrica").loc["max_drawdown"]
        if sh["IC95_inf"] <= 0 <= sh["IC95_sup"]:
            veredicto = "timing no distinguible de exposición fija"
        else:
            veredicto = "timing suma" if sh["diferencia"] > 0 else "timing resta"
        rows.append({
            "variante": label,
            "activos_con_exposicion": int(r["diagnostics"]["dias_evaluables"].gt(0).sum()),
            "inicio": strat.index[0].date(),
            "inicio_5_activos": full_start.date(),
            "anios_evaluables": round(len(strat) / 252, 2),
            "anios_5_activos": round(len(strat.loc[full_start:]) / 252, 2),
            "episodios": ", ".join(_episodes_covered(strat.index)),
            "n_episodios": len(_episodes_covered(strat.index)),
            "exposicion_media": strat["exposure"].mean(),
            "sharpe": own["s"]["Sharpe"], "max_drawdown": own["s"]["Max Drawdown"],
            "dif_sharpe_vs_EW": own["s"]["Sharpe"] - own["ew"]["Sharpe"],
            "dif_mdd_vs_EW": own["s"]["Max Drawdown"] - own["ew"]["Max Drawdown"],
            "dif_sharpe_vs_fija": own["s"]["Sharpe"] - own["fija"]["Sharpe"],
            "IC95_dif_sharpe_vs_fija": f"[{sh['IC95_inf']:.3f}; {sh['IC95_sup']:.3f}]",
            "IC95_dif_mdd_vs_EW": f"[{md['IC95_inf']:.3f}; {md['IC95_sup']:.3f}]",
            "veredicto_timing": veredicto,
            "com_desde": common_start.date(),
            "com_sharpe": com["s"]["Sharpe"], "com_max_drawdown": com["s"]["Max Drawdown"],
            "com_dif_sharpe_vs_EW": com["s"]["Sharpe"] - com["ew"]["Sharpe"],
            "com_dif_mdd_vs_EW": com["s"]["Max Drawdown"] - com["ew"]["Max Drawdown"],
            "com_dif_sharpe_vs_fija": com["s"]["Sharpe"] - com["fija"]["Sharpe"],
        })
    return pd.DataFrame(rows).set_index("variante")


# --------------------------------------------------------------------------
# Robustez en dólares (Entrega #2: "evaluar la robustez en retornos en USD")
# --------------------------------------------------------------------------

USD_PORTFOLIOS = ("Estrategia", "EW 100%", "Exposición fija (media ex post)", "Vol targeting", "MA200")


def usd_view(result: dict, fx_usd_ars: pd.Series, tbill_pct: pd.Series, reps: int = 1000,
             names=USD_PORTFOLIOS) -> pd.DataFrame:
    """
    Las MISMAS carteras (mismas exposiciones y costos) evaluadas en pesos y en dólares.

    No recalibra nada: la señal se calcula con retornos en pesos, como en la
    especificación; solo cambia la moneda en que se miden los resultados.
      - retornos de cada activo: r_usd = (1 + r_ars) / (1 + variación USD/ARS) - 1;
      - la caja rinde el T-bill 3m (base 360), conocido en t-1;
      - Sharpe y Sortino en exceso de esa tasa.
    La fila 'pesos' usa la caución del `result` (result["cash_rate"]).

    Advertencia a declarar: con el tipo de cambio OFICIAL, durante el cepo el
    peso parece menos depreciado de lo que estuvo (CCL/MEP), y los resultados en
    USD quedan sobrestimados. Es una cota, no el retorno dolarizado realizable.
    Vol targeting y MA200 conservan su señal calculada en pesos.
    """
    exposures = build_comparator_exposures(result)
    idx = exposures["Estrategia"].index
    rets_ars = _simple_returns_frame(result, exposures["Estrategia"].columns)
    rets_usd = md.to_usd_returns(rets_ars, md.fx_simple_returns(fx_usd_ars, rets_ars.index))
    cash_usd = md.daily_cash_return(tbill_pct, idx, md.DAY_COUNT_USD)
    cost = result.get("cost_used", result["config"].cost_bps_per_side)
    reb = result["config"].rebalance
    cash_ars = result.get("cash_rate")

    rows, ports_by_ccy = [], {}
    for ccy, rets, cash in (("pesos", rets_ars, cash_ars), ("USD", rets_usd, cash_usd)):
        ports = {n: simulate_portfolio(rets, exposures[n], cost, reb, cash_rate=cash) for n in names}
        ports_by_ccy[ccy] = ports
        for n in names:
            s = summary(ports[n]["net_return"], n, cash)
            rows.append({"moneda": ccy, "cartera": n, "retorno_anual": s["Retorno anualizado"],
                         "volatilidad": s["Volatilidad anualizada"], "sharpe": s["Sharpe"],
                         "max_drawdown": s["Max Drawdown"], "calmar": s["Calmar"],
                         "exposicion_media": ports[n]["exposure"].mean()})
    tab = pd.DataFrame(rows)

    diffs = []
    for ccy, cash in (("pesos", cash_ars), ("USD", cash_usd)):
        ports = ports_by_ccy[ccy]
        for versus in ("EW 100%", "Exposición fija (media ex post)"):
            b = bootstrap_differences(ports["Estrategia"]["net_return"], ports[versus]["net_return"], reps,
                                      risk_free=cash).set_index("métrica")
            diffs.append({"moneda": ccy, "comparador": versus,
                          "dif_sharpe": b.loc["sharpe", "diferencia"],
                          "IC95_sharpe": f"[{b.loc['sharpe', 'IC95_inf']:.3f}; {b.loc['sharpe', 'IC95_sup']:.3f}]",
                          "dif_mdd": b.loc["max_drawdown", "diferencia"],
                          "IC95_mdd": f"[{b.loc['max_drawdown', 'IC95_inf']:.3f}; {b.loc['max_drawdown', 'IC95_sup']:.3f}]"})
    tab.attrs["diferencias"] = pd.DataFrame(diffs)
    return tab


# --------------------------------------------------------------------------
# Piso por drawdown, factorial y activo
# --------------------------------------------------------------------------

def drawdown_floor_grid(result: dict, grid: dict | None = None) -> pd.DataFrame:
    """Grilla completa de (D_max, kappa, E_min) sobre la señal de `result`."""
    grid = grid or DRAWDOWN_GRID
    rows = [{"d_max": None, "kappa": None, "e_min": None, **evaluate_portfolio_variant(result)}]
    for d, k, e in product(grid["d_max"], grid["kappa"], grid["e_min"]):
        ex = apply_drawdown_floor(result["exposures_signal"], result["returns"], d_max=d, kappa=k, e_min=e)
        rows.append({"d_max": d, "kappa": k, "e_min": e, **evaluate_portfolio_variant(result, exposures_pct=ex)})
    return pd.DataFrame(rows)


def factorial_two(result: dict, returns_by_asset: dict, p1: str, v1: list, p2: str, v2: list,
                  price_data: dict | None = None, refit_step: int = 5, verbose: bool = True) -> pd.DataFrame:
    """
    Grilla factorial entre dos parámetros (Sección 5 del .qmd). Los de señal
    reajustan el pipeline; los de cartera se aplican sobre esa corrida.
    """
    from main import PipelineConfig, run_pipeline

    base_kwargs = {**asdict(result["config"]), "windows": dict(result["windows"]), "refit_step": refit_step}
    signal = set(SIGNAL_PARAMS)
    rows = []
    cache = {}
    for a, b in product(v1, v2):
        sig_over = {p: v for p, v in ((p1, a), (p2, b)) if p in signal}
        key = tuple(sorted(sig_over.items()))
        if key not in cache:
            if verbose:
                print(f"  señal {sig_over or 'base'}")
            cache[key] = run_pipeline(returns_by_asset, PipelineConfig(**{**base_kwargs, **sig_over}),
                                      verbose=False, price_data=price_data, cash_rate=result.get("cash_rate"))
        r = cache[key]
        if not r["portfolios"]:
            continue
        ex, cost, reb = None, None, None
        for p, v in ((p1, a), (p2, b)):
            if p == "hysteresis_pp":
                ex = rehysteresis(r, v)
            elif p == "costo":
                cost = _cost_variant(v, r, price_data)
            elif p == "rebalance":
                reb = v
        rows.append({p1: a, p2: b, **evaluate_portfolio_variant(r, exposures_pct=ex, cost=cost, rebalance=reb)})
    return pd.DataFrame(rows)


def by_asset_table(result: dict, cost=None) -> pd.DataFrame:
    """
    Cada activo por separado (Sección 6 del .qmd): estrategia contra buy &
    hold del mismo activo y contra exposición fija a su media, en las fechas
    en que el activo está en la cartera.
    """
    E = build_comparator_exposures(result)
    rets = _simple_returns_frame(result, E["Estrategia"].columns)
    cost = result.get("cost_used", result["config"].cost_bps_per_side) if cost is None else cost
    rf = result.get("cash_rate")
    rows = []
    for t in E["Estrategia"].columns:
        ports = {k: simulate_portfolio(rets[[t]], E[k][[t]], cost if not isinstance(cost, pd.DataFrame) else cost[[t]],
                                       "nunca", cash_rate=rf)
                 for k in ("Estrategia", "EW 100%", "Exposición fija (media ex post)")}
        rows.append({"activo": t, **_metrics_row(ports["Estrategia"], ports["EW 100%"],
                                                 ports["Exposición fija (media ex post)"], rf)})
    return pd.DataFrame(rows).set_index("activo")
