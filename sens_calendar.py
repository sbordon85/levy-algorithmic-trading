"""
sens_calendar.py 

Completa a `sensitivity.py` (Fase 6) y a `robustness.py` con las variaciones de
convención operativa que el OFAT no cubría. No recalibra el NIG ni reajusta el
logit (salvo `refit_step_check`): re-simula carteras sobre el `res` de
`run_pipeline`.

Qué cubre cada pieza
--------------------
1. weekly_rebalance_weekday  Rebalanceo SEMANAL entre activos, un día fijo de la
                             semana (lunes a viernes). Es el ejemplo literal del
                             enunciado. La base del trabajo es mensual, así que
                             se reporta contra ella.
2. execution_lag             Rezago de ejecución de la señal (0, 1, 2, 3, 5
                             ruedas): decidir con información hasta t-1 pero
                             operar L ruedas más tarde.
3. refit_step_check          Frecuencia de reajuste del logit (1 vs 5 vs 10).
                             El OFAT usa 5 y la corrida reportada usa 1: esto
                             verifica que esa diferencia no mueva las
                             conclusiones. Es lo único que re-corre el pipeline.
4. run_all                   Corre 1 y 2 (y 3 si se pasan los datos) y guarda CSV.

Ya cubierto en otros módulos (no se repite acá):
  - día de la semana en que se actualiza la exposición y día del mes del
    rebalanceo mensual: robustness.weekday_update / monthly_rebalance_day;
  - u, h, criterio de W, histéresis, costo, frecuencia de rebalanceo (diario /
    mensual / trimestral / nunca), piso por drawdown, ventanas fijas, USD:
    sensitivity.run_ofat y funciones asociadas.

Lectura: igual que en robustness.py, las filas traen DIFERENCIAS contra EW y
contra la exposición fija. `dif_mdd_vs_EW` > 0 = la estrategia cae menos.

Uso:
    import sens_calendar as sc
    cal = sc.run_all(res)                                   # segundos
    cal = sc.run_all(res, returns_by_asset=returns_by_asset,
                     price_data=price_data, refit_steps=(1, 5, 10))   # lento
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import pandas as pd

import comparators as cmp
from robustness import _row, _simulate, _targets, WEEKDAYS


# --------------------------------------------------------------------------
# 1. Rebalanceo semanal por día de la semana
# --------------------------------------------------------------------------

@contextmanager
def _weekly_rule(weekday: int):
    """
    Registra temporalmente la regla "semanal_wK" en comparators (se restaura
    siempre). Se rebalancea en la primera rueda de la semana con día de la
    semana >= K (si cae feriado el día K, se usa la rueda siguiente de esa
    semana; si K es viernes y ese viernes es feriado, esa semana no rebalancea).
    """
    orig_fn, orig_rules = cmp._is_rebalance_day, cmp.REBALANCE_RULES
    name = f"semanal_w{weekday}"

    def patched(rule, t, prev):
        if rule != name:
            return orig_fn(rule, t, prev)
        if prev is None:
            return True
        if t.isocalendar()[:2] != prev.isocalendar()[:2]:
            return t.weekday() >= weekday
        return prev.weekday() < weekday <= t.weekday()

    cmp._is_rebalance_day = patched
    cmp.REBALANCE_RULES = tuple(orig_rules) + (name,)
    try:
        yield name
    finally:
        cmp._is_rebalance_day, cmp.REBALANCE_RULES = orig_fn, orig_rules


def weekly_rebalance_weekday(res: dict) -> pd.DataFrame:
    """
    Rebalanceo semanal entre activos cada lunes, martes, ..., viernes, contra la
    regla base de la corrida (mensual) y contra el rebalanceo diario.
    La exposición de cada activo sigue cambiando cuando lo decide la señal; lo
    que cambia es cuándo se reponen los pesos iguales entre bolsillos.
    Se espera que el día de la semana no mueva los resultados (Mayer-Krebs,
    cap. 7.1: el día de la semana no es fuente de alfa).
    """
    t, r = _targets(res)
    rf = res.get("cash_rate")
    base = res["config"].rebalance
    rows = [{"rebalanceo": f"{base} (base)", **_row(_simulate(res, t, r), rf)},
            {"rebalanceo": "diario", **_row(_simulate(res, t, r, rebalance="diario"), rf)}]
    for k, name in WEEKDAYS.items():
        with _weekly_rule(k) as rule:
            ports = _simulate(res, t, r, rebalance=rule)
        rows.append({"rebalanceo": f"semanal {name}", **_row(ports, rf)})
    out = pd.DataFrame(rows).set_index("rebalanceo")
    wk = out.index.str.startswith("semanal")
    out["d_sharpe_vs_base"] = out["sharpe"] - out.iloc[0]["sharpe"]
    out["d_mdd_vs_base"] = out["max_drawdown"] - out.iloc[0]["max_drawdown"]
    out.attrs["resumen_semanal"] = pd.Series({
        "sharpe_min": out.loc[wk, "sharpe"].min(), "sharpe_max": out.loc[wk, "sharpe"].max(),
        "rango_sharpe": out.loc[wk, "sharpe"].max() - out.loc[wk, "sharpe"].min(),
        "mdd_min": out.loc[wk, "max_drawdown"].min(), "mdd_max": out.loc[wk, "max_drawdown"].max(),
        "dif_mdd_vs_EW_min": out.loc[wk, "dif_mdd_vs_EW"].min(),
    })
    return out


# --------------------------------------------------------------------------
# 2. Rezago de ejecución
# --------------------------------------------------------------------------

def execution_lag(res: dict, lags=(0, 1, 2, 3, 5)) -> pd.DataFrame:
    """
    La señal se calcula con información hasta t-1 y rige desde el retorno de t
    (lag 0, base). Con lag L, la exposición decidida se ejecuta L ruedas más
    tarde: la estrategia opera con una señal "vieja". Mide cuánto del resultado
    depende de poder ejecutar justo al cierre siguiente. Las primeras L ruedas
    de cada activo usan su primera exposición (no hay señal anterior).

    El comparador de exposición fija se recalcula con la media de la exposición
    rezagada; EW no cambia.
    """
    rows = []
    for L in lags:
        if L == 0:
            ex = None
        else:
            ex = {}
            for t_, e in res["exposures"].items():
                v = e.dropna()
                sh = v.shift(L).fillna(v.iloc[0]) if len(v) else v
                ex[t_] = sh.reindex(e.index)
        t, r = _targets(res, ex)
        rows.append({"rezago_ruedas": L, **_row(_simulate(res, t, r), res.get("cash_rate"))})
    out = pd.DataFrame(rows).set_index("rezago_ruedas")
    out["d_sharpe_vs_lag0"] = out["sharpe"] - out.iloc[0]["sharpe"]
    out["d_mdd_vs_lag0"] = out["max_drawdown"] - out.iloc[0]["max_drawdown"]
    return out


# --------------------------------------------------------------------------
# 3. Frecuencia de reajuste del logit
# --------------------------------------------------------------------------

def refit_step_check(res: dict, returns_by_asset: dict, price_data: dict | None = None,
                     steps=(1, 5, 10), verbose: bool = True) -> pd.DataFrame:
    """
    Re-corre el pipeline con distinto `refit_step` (reajuste del logit cada N
    ruedas), manteniendo W fijos y el NIG desde caché. `refit_step=1` es el
    caso de la corrida reportada (puede tardar bastante: es el más lento).
    """
    from main import PipelineConfig, run_pipeline

    base_kwargs = {**asdict(res["config"]), "windows": dict(res["windows"])}
    rows = []
    for s in steps:
        t0 = time.time()
        r = run_pipeline(returns_by_asset, PipelineConfig(**{**base_kwargs, "refit_step": int(s)}),
                         verbose=False, price_data=price_data, cash_rate=res.get("cash_rate"))
        if verbose:
            print(f"  refit_step={s}: {time.time() - t0:.0f} s")
        if not r["portfolios"]:
            continue
        rows.append({"refit_step": s, **_row(
            {k: r["portfolios"][k] for k in ("Estrategia", "EW 100%", "Exposición fija (media ex post)")},
            res.get("cash_rate"))})
    out = pd.DataFrame(rows).set_index("refit_step")
    if len(out):
        out["d_sharpe_vs_primero"] = out["sharpe"] - out.iloc[0]["sharpe"]
        out["d_mdd_vs_primero"] = out["max_drawdown"] - out.iloc[0]["max_drawdown"]
    return out


# --------------------------------------------------------------------------
# Todo junto
# --------------------------------------------------------------------------

def run_all(res: dict, returns_by_asset: dict | None = None, price_data: dict | None = None,
            refit_steps=None, prefix: str = "sens_cal_", save: bool = True, verbose: bool = True) -> dict:
    """Corre 1 y 2; 3 solo si se pasan `returns_by_asset` y `refit_steps`."""
    log = print if verbose else (lambda *a, **k: None)
    out: dict = {}

    def step(name, fn):
        t0 = time.time()
        out[name] = fn()
        log(f"  {name}: {time.time() - t0:.0f} s")

    log("Sensibilidad (punto 2, convenciones operativas)")
    step("rebalanceo_semanal", lambda: weekly_rebalance_weekday(res))
    step("rezago_ejecucion", lambda: execution_lag(res))
    if returns_by_asset is not None and refit_steps:
        step("refit_step", lambda: refit_step_check(res, returns_by_asset, price_data, refit_steps, verbose))
    if save:
        for name, df in out.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(f"{prefix}{name}.csv")
    return out
