"""
robustness.py  (Entrega #4, punto 1: "Robustez del research")

Pregunta que responde: ¿los resultados dependen demasiado de una FECHA, un
PARÁMETRO, un PERÍODO o una CONVENCIÓN del backtest?

Complementa a `sensitivity.py` (Fase 6), que ya cubre el OFAT de parámetros, la
grilla del piso por drawdown, las ventanas fijas y la vista en USD. Acá solo
está lo que faltaba. NO recalibra el NIG ni reajusta el logit: trabaja sobre
el `res` que devuelve `run_pipeline` (re-simula carteras con el motor de
`comparators.simulate_portfolio`), así que corre en segundos o minutos.

Contenido
---------
 1. date_sweep          Barrido de fecha de inicio (o de fin): ¿la ventaja
                        contra EW / exposición fija depende de cuándo se
                        arranca?
 2. yearly_table        Resultado por año calendario.
 3. leave_one_asset_out Cartera sin cada activo (en particular sin BYMA).
 4. weekday_update      Día de la semana en que se actualiza la exposición
                        (lunes a viernes). Análogo del ejemplo del enunciado.
 5. monthly_rebalance_day  Día del mes del rebalanceo entre activos.
 6. random_timing_mc    Monte Carlo de timing aleatorio (Mayer-Krebs, cap. 9):
                        desplazamientos circulares de la exposición real. Mantiene
                        el nivel de exposición y el turnover, y rompe solo la
                        alineación con los retornos.
 7. dsr_vs_n            Deflated Sharpe según la cantidad TOTAL de pruebas.
 8. param_favorable_share  % de variantes del OFAT favorables (Kaufman, cap. 21).
 9. logit_stability     Estabilidad de las pendientes del logit walk-forward.
10. exposure_changes    Cantidad de cambios de exposición por activo.
11. run_all             Corre todo y guarda CSV.

Convención de lectura
---------------------
Todas las filas reportan, además de niveles, DIFERENCIAS contra los
comparadores de la misma corrida (EW 100% y exposición fija a la media). El
Sharpe absoluto cambia mucho con la muestra; las diferencias son lo
comparable. `dif_mdd_*` > 0 significa que la estrategia cae menos.

Uso (desde el notebook, después de correr `res`):

    import robustness as rb
    rob = rb.run_all(res, price_data=price_data, ofat=ofat,
                     grid_returns=grilla, chosen=ELEGIDA, risk_free=CAJA)
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

import comparators as cmp
from comparators import _simple_returns_frame, active_mask_from_result, simulate_portfolio
from performance_metrics import summary
from protection import deflated_sharpe_ratio

STRAT = "Estrategia"
EW = "EW 100%"
FIJA = "Exposición fija (media ex post)"
TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Utilidades internas
# --------------------------------------------------------------------------

def _ctx(res: dict):
    """Costo, regla de rebalanceo y tasa de caja usados en la corrida base."""
    cfg = res["config"]
    return res.get("cost_used", cfg.cost_bps_per_side), cfg.rebalance, res.get("cash_rate")


def _targets(res: dict, exposures_pct: dict | None = None, assets: list[str] | None = None):
    """
    Exposiciones objetivo (fracción) de Estrategia, EW 100% y exposición fija,
    sobre la misma máscara activo/fecha de la estrategia que usa
    comparators.build_comparator_exposures. Admite otra exposición
    (`exposures_pct`, en [0, 100]) y un subconjunto de activos.
    """
    mask = active_mask_from_result(res)
    cols = list(mask.columns) if assets is None else [a for a in assets if a in mask.columns]
    mask = mask[cols]
    mask = mask[mask.any(axis=1)]
    expo = res["exposures"] if exposures_pct is None else exposures_pct
    strat = (pd.DataFrame({t: expo[t] for t in cols}).reindex(mask.index) / 100.0).where(mask)
    ones = pd.DataFrame(1.0, index=mask.index, columns=cols).where(mask)
    fija = ones * strat.mean()
    rets = _simple_returns_frame(res, cols).reindex(mask.index)
    return {STRAT: strat, EW: ones, FIJA: fija}, rets


def _simulate(res: dict, targets: dict, rets: pd.DataFrame, rebalance: str | None = None,
              names=(STRAT, EW, FIJA)) -> dict[str, pd.DataFrame]:
    cost, reb, rf = _ctx(res)
    reb = rebalance or reb
    return {n: simulate_portfolio(rets, targets[n], cost, reb, cash_rate=rf) for n in names}


def _row(ports: dict, rf) -> dict:
    """Niveles y diferencias de la estrategia contra EW y exposición fija."""
    s, e, f = (summary(ports[k]["net_return"], risk_free_rate=rf) for k in (STRAT, EW, FIJA))
    st = ports[STRAT]
    return {
        "dias": len(st), "anios": round(len(st) / TRADING_DAYS, 2),
        "sharpe": s["Sharpe"], "sharpe_EW": e["Sharpe"],
        "retorno_anual": s["Retorno anualizado"], "max_drawdown": s["Max Drawdown"],
        "mdd_EW": e["Max Drawdown"],
        "exposicion_media": st["exposure"].mean(),
        "turnover_anual": st["turnover"].sum() / (len(st) / TRADING_DAYS),
        "dif_sharpe_vs_EW": s["Sharpe"] - e["Sharpe"],
        "dif_mdd_vs_EW": s["Max Drawdown"] - e["Max Drawdown"],
        "dif_sharpe_vs_fija": s["Sharpe"] - f["Sharpe"],
        "dif_mdd_vs_fija": s["Max Drawdown"] - f["Max Drawdown"],
    }


def _variant(res: dict, exposures_pct: dict | None = None, assets: list[str] | None = None,
             rebalance: str | None = None) -> dict:
    t, r = _targets(res, exposures_pct, assets)
    return _row(_simulate(res, t, r, rebalance), res.get("cash_rate"))


def _base_returns(res: dict) -> pd.DataFrame:
    """Retornos netos diarios de las tres carteras de la corrida base."""
    p = res["portfolios"]
    return pd.DataFrame({k: p[k]["net_return"] for k in (STRAT, EW, FIJA)}).dropna()


def _diff_row(slice_: pd.DataFrame, rf) -> dict:
    rfs = None if rf is None else rf.reindex(slice_.index).fillna(0.0)
    s, e, f = (summary(slice_[k], risk_free_rate=rfs) for k in (STRAT, EW, FIJA))
    return {
        "dias": len(slice_), "anios": round(len(slice_) / TRADING_DAYS, 2),
        "sharpe": s["Sharpe"], "sharpe_EW": e["Sharpe"],
        "max_drawdown": s["Max Drawdown"], "mdd_EW": e["Max Drawdown"],
        "dif_sharpe_vs_EW": s["Sharpe"] - e["Sharpe"],
        "dif_mdd_vs_EW": s["Max Drawdown"] - e["Max Drawdown"],
        "dif_sharpe_vs_fija": s["Sharpe"] - f["Sharpe"],
        "dif_mdd_vs_fija": s["Max Drawdown"] - f["Max Drawdown"],
    }


# --------------------------------------------------------------------------
# 1. Barrido de fecha de inicio / fin
# --------------------------------------------------------------------------

def _first_full_date(res: dict) -> pd.Timestamp:
    n = res["portfolios"][STRAT]["n_assets"]
    return n[n == n.max()].index[0]


def date_sweep(res: dict, mode: str = "start", step_months: int = 3, min_years: float = 2.0,
               start_from="full") -> pd.DataFrame:
    """
    Recalcula las métricas contra EW y exposición fija moviendo la fecha de
    inicio (mode="start": inicio cada `step_months`, hasta que queden
    `min_years`) o la de fin (mode="end": fin cada `step_months`, desde
    `min_years` después del inicio).

    start_from: "full" = primera fecha con todos los activos en la cartera
    (evita mezclar composiciones distintas; recomendado), "first" = primera
    fecha de la cartera, o una fecha ("2023-01-02").

    Qué mirar: si `dif_mdd_vs_EW` y `dif_sharpe_vs_*` mantienen el signo en
    casi todas las filas, el resultado no depende de la fecha. Las ventanas se
    solapan: no son observaciones independientes, es un diagnóstico de
    sensibilidad y no una prueba. Resumen en `df.attrs["resumen"]`.
    """
    if mode not in ("start", "end"):
        raise ValueError("mode debe ser 'start' o 'end'")
    r = _base_returns(res)
    rf = res.get("cash_rate")
    if start_from == "full":
        t0 = _first_full_date(res)
    elif start_from == "first":
        t0 = r.index[0]
    else:
        t0 = pd.Timestamp(start_from)
    r = r.loc[t0:]
    t_end = r.index[-1]
    min_off = pd.DateOffset(months=int(round(min_years * 12)))
    step = pd.DateOffset(months=step_months)

    rows = []
    if mode == "start":
        d = r.index[0]
        while d + min_off <= t_end:
            sl = r.loc[d:]
            rows.append({"inicio": sl.index[0].date(), "fin": sl.index[-1].date(), **_diff_row(sl, rf)})
            d = d + step
    else:
        d = r.index[0] + min_off
        ends = []
        while d <= t_end:
            ends.append(d)
            d = d + step
        ends.append(t_end)
        for e in dict.fromkeys(ends):
            sl = r.loc[:e]
            rows.append({"inicio": sl.index[0].date(), "fin": sl.index[-1].date(), **_diff_row(sl, rf)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs["resumen"] = pd.Series({
            "ventanas": len(out),
            "% con dif_mdd_vs_EW > 0": 100 * (out["dif_mdd_vs_EW"] > 0).mean(),
            "% con dif_sharpe_vs_EW >= 0": 100 * (out["dif_sharpe_vs_EW"] >= 0).mean(),
            "% con dif_sharpe_vs_fija >= 0": 100 * (out["dif_sharpe_vs_fija"] >= 0).mean(),
            "dif_mdd_vs_EW (mín / mediana / máx)": f"{out['dif_mdd_vs_EW'].min():.3f} / "
                                                    f"{out['dif_mdd_vs_EW'].median():.3f} / {out['dif_mdd_vs_EW'].max():.3f}",
            "dif_sharpe_vs_EW (mín / mediana / máx)": f"{out['dif_sharpe_vs_EW'].min():.3f} / "
                                                       f"{out['dif_sharpe_vs_EW'].median():.3f} / {out['dif_sharpe_vs_EW'].max():.3f}",
            "Sharpe EW (mín / máx)": f"{out['sharpe_EW'].min():.2f} / {out['sharpe_EW'].max():.2f}",
        })
    return out


# --------------------------------------------------------------------------
# 2. Año calendario
# --------------------------------------------------------------------------

def yearly_table(res: dict, min_days: int = 60) -> pd.DataFrame:
    """Retorno, Sharpe y drawdown por año calendario, con diferencias contra EW y fija."""
    r = _base_returns(res)
    rf = res.get("cash_rate")
    rows = []
    for y, sl in r.groupby(r.index.year):
        row = {"anio": int(y), "dias": len(sl),
               "retorno_estrategia": (1 + sl[STRAT]).prod() - 1,
               "retorno_EW": (1 + sl[EW]).prod() - 1}
        row["dif_retorno_vs_EW"] = row["retorno_estrategia"] - row["retorno_EW"]
        if len(sl) >= min_days:
            row.update({k: v for k, v in _diff_row(sl, rf).items() if k not in ("dias", "anios")})
        rows.append(row)
    return pd.DataFrame(rows).set_index("anio")


# --------------------------------------------------------------------------
# 3. Leave-one-asset-out
# --------------------------------------------------------------------------

def leave_one_asset_out(res: dict) -> pd.DataFrame:
    """
    Cartera con todos los activos y sin cada uno. Cada fila compara contra el
    EW y la exposición fija de los MISMOS activos. Si la ventaja sobrevive sin
    BYMA, no depende de un solo activo.
    """
    assets = list(active_mask_from_result(res).columns)
    rows = [{"variante": "todos", **_variant(res)}]
    for a in assets:
        keep = [x for x in assets if x != a]
        if keep:
            rows.append({"variante": f"sin {a}", **_variant(res, assets=keep)})
    return pd.DataFrame(rows).set_index("variante")


# --------------------------------------------------------------------------
# 4. Día de la semana de actualización de la exposición
# --------------------------------------------------------------------------

WEEKDAYS = {0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves", 4: "viernes"}


def _hold_weekly(e: pd.Series, weekday: int) -> pd.Series:
    """
    La exposición solo se actualiza el primer día hábil de cada semana con
    día de la semana >= `weekday` (si el feriado cae justo ese día, se usa el
    siguiente hábil; si no hay ninguno en la semana, se mantiene). El resto
    de los días se mantiene la exposición vigente. La primera fecha válida
    siempre actualiza.
    """
    v = e.dropna()
    if v.empty:
        return e.copy()
    iso = v.index.isocalendar()
    key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    cand = np.asarray(v.index.weekday >= weekday)
    df = pd.DataFrame({"k": key, "c": cand}, index=v.index)
    df["n"] = df.groupby("k")["c"].cumsum()
    upd = (df["c"] & (df["n"] == 1)).to_numpy().copy()
    upd[0] = True
    held = v.where(upd).ffill()
    return held.reindex(e.index)


def weekday_update(res: dict) -> pd.DataFrame:
    """
    ¿Cambia el resultado si la exposición se actualiza solo un día fijo de la
    semana? La fila "diaria (base)" es la corrida original. Mayer-Krebs
    (cap. 7.1): el día de la semana no es fuente de alfa, así que se espera
    NO ver diferencias significativas entre filas.
    """
    rows = [{"actualiza": "diaria (base)", **_variant(res)}]
    for k, name in WEEKDAYS.items():
        ex = {t: _hold_weekly(e, k) for t, e in res["exposures"].items()}
        rows.append({"actualiza": name, **_variant(res, exposures_pct=ex)})
    out = pd.DataFrame(rows).set_index("actualiza")
    base = out.loc["diaria (base)"]
    out["d_sharpe_vs_base"] = out["sharpe"] - base["sharpe"]
    out["d_mdd_vs_base"] = out["max_drawdown"] - base["max_drawdown"]
    return out


# --------------------------------------------------------------------------
# 5. Día del mes del rebalanceo entre activos
# --------------------------------------------------------------------------

@contextmanager
def _monthly_day_rule(day: int):
    """Registra temporalmente la regla "mensual_dK" en comparators (se restaura siempre)."""
    orig_fn, orig_rules = cmp._is_rebalance_day, cmp.REBALANCE_RULES
    name = f"mensual_d{day}"

    def patched(rule, t, prev):
        if rule != name:
            return orig_fn(rule, t, prev)
        if prev is None:
            return True
        if (t.year, t.month) != (prev.year, prev.month):
            return t.day >= day
        return prev.day < day <= t.day

    cmp._is_rebalance_day = patched
    cmp.REBALANCE_RULES = tuple(orig_rules) + (name,)
    try:
        yield name
    finally:
        cmp._is_rebalance_day, cmp.REBALANCE_RULES = orig_fn, orig_rules


def monthly_rebalance_day(res: dict, days=(1, 5, 10, 15, 20)) -> pd.DataFrame:
    """
    Rebalanceo mensual entre activos en el primer día hábil con día del mes
    >= K. K=1 es la regla base (se opera al cierre del último día del mes
    anterior, equivalente a "rebalancear a fin de mes").
    """
    t, r = _targets(res)
    rows = []
    for d in days:
        with _monthly_day_rule(d) as name:
            ports = _simulate(res, t, r, rebalance=name)
        rows.append({"dia_del_mes": d, **_row(ports, res.get("cash_rate"))})
    out = pd.DataFrame(rows).set_index("dia_del_mes")
    ref = out.loc[days[0]]
    out["d_sharpe_vs_primero"] = out["sharpe"] - ref["sharpe"]
    return out


# --------------------------------------------------------------------------
# 6. Monte Carlo de timing aleatorio
# --------------------------------------------------------------------------

def random_timing_mc(res: dict, n: int = 500, seed: int = 0, verbose: bool = False) -> dict:
    """
    Placebo de timing por desplazamiento circular (Mayer-Krebs, cap. 9, con la
    variante del desplazamiento en lugar de sorteo activo/caja).

    En cada réplica, la serie de exposición de CADA activo se desplaza
    circularmente un número aleatorio de ruedas dentro de su propio período
    activo. Se conservan exactamente: la distribución de niveles (exposición
    media), el número de cambios y la autocorrelación de la exposición. Se
    rompe solo la alineación con los retornos. Cada réplica se simula con el
    mismo motor, costos, rebalanceo y caja que la estrategia.

    Hipótesis nula: el timing de la estrategia no es mejor que uno aleatorio
    con el mismo nivel de exposición. `p_*` = proporción de réplicas
    IGUAL O MEJORES que la estrategia observada (chico = el timing aporta).

    Limitación: el desplazamiento independiente por activo rompe también la
    coincidencia de las bajas de exposición entre activos.

    Returns dict con "observado", "placebo" (DataFrame n x métricas) y "resumen".
    """
    rng = np.random.default_rng(seed)
    cost, reb, rf = _ctx(res)
    targets, rets = _targets(res)
    strat = targets[STRAT]
    cols, idx = list(strat.columns), strat.index
    S = strat.to_numpy()
    pos = [np.flatnonzero(~np.isnan(S[:, j])) for j in range(S.shape[1])]

    def metrics(port):
        s = summary(port["net_return"], risk_free_rate=rf)
        return {"sharpe": s["Sharpe"], "retorno_anual": s["Retorno anualizado"],
                "max_drawdown": s["Max Drawdown"], "calmar": s["Calmar"]}

    obs = metrics(simulate_portfolio(rets, strat, cost, reb, cash_rate=rf))
    ew = metrics(simulate_portfolio(rets, targets[EW], cost, reb, cash_rate=rf))

    draws = []
    t0 = time.time()
    for i in range(n):
        M = np.full_like(S, np.nan)
        for j, p in enumerate(pos):
            if len(p) < 2:
                M[p, j] = S[p, j]
                continue
            M[p, j] = np.roll(S[p, j], int(rng.integers(1, len(p))))
        draws.append(metrics(simulate_portfolio(rets, pd.DataFrame(M, index=idx, columns=cols),
                                                cost, reb, cash_rate=rf)))
        if verbose and (i + 1) % 100 == 0:
            print(f"  placebo {i + 1}/{n} ({time.time() - t0:.0f} s)")
    plc = pd.DataFrame(draws)

    resumen = pd.DataFrame({
        "observado": pd.Series(obs),
        "EW 100%": pd.Series(ew),
        "placebo_mediana": plc.median(),
        "placebo_p5": plc.quantile(0.05),
        "placebo_p95": plc.quantile(0.95),
        "p_igual_o_mejor": pd.Series({k: float((plc[k] >= obs[k]).mean()) for k in obs}),
    })
    return {"observado": obs, "placebo": plc, "resumen": resumen}


# --------------------------------------------------------------------------
# 7. Deflated Sharpe según la cantidad total de pruebas
# --------------------------------------------------------------------------

def dsr_vs_n(grid_returns: pd.DataFrame, chosen: str, n_values=(None, 50, 75, 100, 150, 200),
             min_days: int = 2 * TRADING_DAYS, risk_free: pd.Series | None = None) -> pd.DataFrame:
    """
    DSR de la configuración elegida variando el N de pruebas declarado.

    Replica el cálculo de protection.overfitting_summary: Sharpe de las
    configuraciones con al menos `min_days` de historia, medidos sobre las
    FECHAS COMUNES (si cada una se midiera en su propio período, la dispersión
    mezclaría período y configuración e inflaría SR0), y retornos en exceso de
    `risk_free`. La fila N = None usa el N de la grilla (todas las columnas) y
    debe coincidir con el DSR del notebook. Las demás filas suben N manteniendo
    la dispersión de la grilla, que es una aproximación a la de todas las
    pruebas (OFAT, ventanas fijas, piso por drawdown, costos...).
    Mayer-Krebs (cap. 9.5): hay que registrar y declarar el N total.

    grid_returns: DataFrame de retornos netos diarios por configuración
    (config_grid_returns*.csv); chosen: nombre de la columna elegida.
    """
    g = grid_returns
    if risk_free is not None:
        g = g.sub(risk_free.reindex(g.index).fillna(0.0), axis=0)
    n_grid = g.shape[1]
    long_enough = [c for c in g if g[c].notna().sum() >= min_days]
    common = g[long_enough].dropna()
    if chosen not in common or len(common) <= 20 or len(long_enough) < 2:
        raise ValueError("No hay suficientes configuraciones con fechas comunes para calcular el DSR.")
    sr_common = (common.mean() / common.std(ddof=1)).to_numpy()
    rows = []
    for n in n_values:
        n_eff = n_grid if n is None else int(n)
        d = deflated_sharpe_ratio(common[chosen], sr_common, n_trials=n_eff)
        rows.append({"N_pruebas": n_eff, "T": d["T"], "sharpe_anual": d["sharpe_anual"],
                     "SR0_anual": d["SR0_anual"], "DSR": d["DSR"]})
    out = pd.DataFrame(rows).set_index("N_pruebas")
    out.attrs["fechas_comunes"] = f"{common.index[0].date()} → {common.index[-1].date()}"
    out.attrs["configs_en_calculo"] = len(long_enough)
    return out


# --------------------------------------------------------------------------
# 8. Porcentaje de variantes favorables (OFAT)
# --------------------------------------------------------------------------

def param_favorable_share(ofat: pd.DataFrame) -> pd.DataFrame:
    """
    Kaufman (cap. 21, "% Prof Tests"): en lugar de mirar la mejor variante,
    qué porcentaje de las variantes del OFAT mantiene la conclusión. Excluye la
    fila 'base'. Por parámetro y total.
    """
    d = ofat[ofat["parametro"] != "base"].copy()
    d["mdd_mejor_que_EW"] = d["dif_mdd_vs_EW"] > 0
    d["sharpe_no_peor_que_EW"] = d["dif_sharpe_vs_EW"] >= 0
    d["sharpe_no_peor_que_fija"] = d["dif_sharpe_vs_fija"] >= 0
    cols = ["mdd_mejor_que_EW", "sharpe_no_peor_que_EW", "sharpe_no_peor_que_fija"]
    g = (100 * d.groupby("parametro")[cols].mean()).round(1)
    g.insert(0, "variantes", d.groupby("parametro").size())
    total = pd.DataFrame([{"variantes": len(d), **(100 * d[cols].mean()).round(1).to_dict()}], index=["TOTAL"])
    return pd.concat([g, total])


# --------------------------------------------------------------------------
# 9. Estabilidad del logit
# --------------------------------------------------------------------------

def logit_stability(res: dict) -> pd.DataFrame:
    """
    Estabilidad de las pendientes del logit walk-forward (equivalente a mirar
    la estabilidad de los parámetros entre reajustes, Mayer-Krebs cap. 7.3).
    Muchos cambios de signo o rango amplio = señal inestable en el tiempo.
    """
    fit = res.get("fit")
    if fit is None or fit.history is None or fit.history.empty:
        return pd.DataFrame()
    h = fit.history
    rows = {}
    for c in ("z_lambda", "z_dlambda", "z_lambda:z_dlambda"):
        if c not in h:
            continue
        x = h[c].dropna()
        sg = np.sign(x.to_numpy())
        rows[c] = {"ajustes": len(x), "primero": x.iloc[0], "ultimo": x.iloc[-1],
                   "mediana": x.median(), "min": x.min(), "max": x.max(), "desvio": x.std(),
                   "% positivo": 100 * (x > 0).mean(), "cambios_de_signo": int((sg[1:] != sg[:-1]).sum())}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# 10. Cambios de exposición
# --------------------------------------------------------------------------

def exposure_changes(res: dict) -> pd.DataFrame:
    """Cuántas veces cambió la exposición por activo (Kaufman: ¿hay suficientes operaciones?)."""
    d = res["diagnostics"][["dias_evaluables", "anios_evaluables", "cambios_exposicion",
                            "exp_min", "exp_media", "exp_max"]].copy()
    d["cambios_por_anio"] = (d["cambios_exposicion"] / d["anios_evaluables"]).round(2)
    d.loc["TOTAL"] = [np.nan, np.nan, d["cambios_exposicion"].sum(), np.nan, np.nan, np.nan, np.nan]
    return d


# --------------------------------------------------------------------------
# Todo junto
# --------------------------------------------------------------------------

def run_all(res: dict, price_data: dict | None = None, ofat: pd.DataFrame | None = None,
            grid_returns=None, chosen=None, risk_free=None, n_mc: int = 500, prefix: str = "robustez_", save: bool = True,
            verbose: bool = True) -> dict:
    """
    Corre todo lo que se pueda con lo que se pase. `ofat` habilita
    param_favorable_share; `grid_returns` + `chosen` habilitan dsr_vs_n. `price_data` se
    acepta por compatibilidad (el costo ya viene en res["cost_used"]).
    Guarda un CSV por tabla con el prefijo `prefix`.
    """
    log = print if verbose else (lambda *a, **k: None)
    out: dict = {}

    def step(name, fn):
        t0 = time.time()
        out[name] = fn()
        log(f"  {name}: {time.time() - t0:.0f} s")

    log("Robustez (punto 1)")
    step("cambios_exposicion", lambda: exposure_changes(res))
    step("barrido_inicio", lambda: date_sweep(res, "start"))
    step("barrido_fin", lambda: date_sweep(res, "end"))
    step("por_anio", lambda: yearly_table(res))
    step("sin_activo", lambda: leave_one_asset_out(res))
    step("dia_semana", lambda: weekday_update(res))
    step("dia_mes_rebalanceo", lambda: monthly_rebalance_day(res))
    step("estabilidad_logit", lambda: logit_stability(res))
    mc = {}

    def _mc():
        mc.update(random_timing_mc(res, n=n_mc, verbose=verbose))
        return mc["resumen"]

    step("timing_aleatorio", _mc)
    if ofat is not None:
        step("variantes_favorables", lambda: param_favorable_share(ofat))
    if grid_returns is not None and chosen is not None:
        step("dsr_segun_N", lambda: dsr_vs_n(grid_returns, chosen, risk_free=risk_free))

    if save:
        for name, df in out.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(f"{prefix}{name}.csv")
        if mc:
            mc["placebo"].to_csv(f"{prefix}timing_aleatorio_replicas.csv", index=False)
    out["_placebo"] = mc.get("placebo")
    return out
