"""
main.py

Organiza el pipeline de punta a punta sobre el universo de 5 activos:

  1. Descarga y limpieza por activo (data_loader.py)
  2. Ventana W_i por activo (window_selection.py) -- criterio configurable
  3-4. Calibración NIG walk-forward + estimadores (levy_estimators.py)
  5-7. Eventos, logit pooled Point-In-Time, exposición con histéresis
       (signal_generator.py)
  8. Backtest, overlay equal-weight y métricas (backtest_engine.py,
     performance_metrics.py)

Uso:
    python main.py                     # corrida con PipelineConfig() por defecto

Piezas reutilizables (notebook, Fase 6):
    run_pipeline(returns_by_asset, config)   -> dict con todos los resultados
    signal_diagnostics(result)               -> cobertura y "señal viva" por activo
    compare_window_criteria(returns_by_asset, ...) -> diagnóstico para elegir W

El logit se entrena point-in-time (solo
eventos con horizonte cerrado antes de cada decision). 

Requiere acceso a Yahoo Finance (yfinance).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from data_loader import load_benchmark, load_price_data
from window_selection import (
    DEFAULT_WINDOW_COMBINE,
    DEFAULT_WINDOW_CRITERION,
    EPISODE_CATALOG,
    compute_asset_window,
)
from levy_estimators import compute_rolling_estimators
from signal_generator import (
    DEFAULT_EVENT_DEFINITION,
    DEFAULT_TAU_METHOD,
    HORIZON_H,
    HYSTERESIS_PP,
    generate_exposures,
)
from backtest_engine import (
    COST_BPS_PER_SIDE_BY_ASSET,
    cost_table,
    equal_weight_benchmark,
    run_multi_asset_backtest,
)
from performance_metrics import compare
from liquidity import (
    ADV_CAP_PCT,
    AUM_ARS,
    SPREAD_WINDOW,
    adv_ars,
    adv_diagnostics,
    apply_drawdown_floor,
    cost_for_mode,
)
from comparators import (
    DEFAULT_REBALANCE,
    comparison_table,
    metrics_by_period,
    run_comparators,
)

TICKERS = ["GGAL.BA", "PAMP.BA", "YPFD.BA", "BMA.BA", "BYMA.BA"]

# 1 = recalibración diaria del NIG. Para la
# comparación de criterios de W (varias corridas) conviene 5; para la
# corrida que se reporta, 1.
RECALIBRATION_STEP = 1


@dataclass
class PipelineConfig:
    window_criterion: str = DEFAULT_WINDOW_CRITERION   # parámetros | sin_mu | estandarizado | estimadores
    window_combine: str = DEFAULT_WINDOW_COMBINE       # suma | max
    recalibration_step: int = RECALIBRATION_STEP
    event_definition: str = DEFAULT_EVENT_DEFINITION   # diaria | horizonte
    tau_method: str = DEFAULT_TAU_METHOD               # excedencias | modelo
    walk_forward: bool = True                          # False = ajuste único con look-ahead
    refit_step: int = 1                                # reajuste del logit (1 = diario)
    horizon: int = HORIZON_H
    u_multiplier: float = 2.0                          # umbral de salto u = k * sigma_hat
    hysteresis_pp: float = HYSTERESIS_PP
    rebalance: str = DEFAULT_REBALANCE                 # diario | mensual | trimestral | nunca (entre activos)
    cost_mode: str = "variable"                        # medio | p95 | variable (Corwin-Schultz diario, Fase 5)
    spread_window: int = SPREAD_WINDOW                 # suavizado del spread diario (ruedas)
    aum_ars: float = AUM_ARS                           # AUM inicial hipotético para el chequeo de ADV
    adv_cap_pct: float = ADV_CAP_PCT                   # tope de participación en el ADV_20
    drawdown_floor: dict | None = None                 # p. ej. liquidity.DEFAULT_DRAWDOWN_FLOOR; None = sin piso
    cost_bps_per_side: dict | float = field(default_factory=lambda: dict(COST_BPS_PER_SIDE_BY_ASSET))
    cache_dir: str | None = "cache_nig"                # None = sin caché
    windows: dict | None = None                        # W_i fijos (salta window_selection)

    def label(self) -> str:
        return (f"W:{self.window_criterion}/{self.window_combine} | evento:{self.event_definition} "
                f"| tau:{self.tau_method} | {'walk-forward' if self.walk_forward else 'in-sample'} "
                f"| rebalanceo:{self.rebalance} | costo:{self.cost_mode}"
                f"{' | piso DD' if self.drawdown_floor else ''}")


def run_pipeline(returns_by_asset: dict[str, pd.Series], config: PipelineConfig | None = None,
                 verbose: bool = True, merval_log_returns: pd.Series | None = None,
                 price_data: dict[str, pd.DataFrame] | None = None,
                 cash_rate: pd.Series | None = None) -> dict:
    """
    Corre los pasos 2-8 para un set de retornos (log) y una configuración.

    Resultado principal (Fase 3): `portfolios` y `comparison` salen del motor
    de tenencias de comparators.py, con compra inicial y rebalanceo con
    costos (`config.rebalance`) para la estrategia y todos los comparadores.

    Fase 5: con `price_data` (salida de load_price_data, con High/Low/Close/
    Volume) el costo puede ser variable (`config.cost_mode="variable"`) y se
    agrega el diagnóstico de participación en el ADV (`result["liquidity"]`).
    Sin `price_data`, el costo variable cae al costo medio constante.

    Variables macro: `cash_rate` es el rendimiento diario de la
    caja (macro_data.daily_cash_return, caución a 1 día en pesos). Con él, la
    caja de la cartera rinde intereses y Sharpe/Sortino se calculan en exceso
    de la tasa. Sin él (None), caja al 0% y tasa libre de riesgo 0.
    """
    config = config or PipelineConfig()
    log = print if verbose else (lambda *a, **k: None)

    # --- Paso 2: ventanas ------------------------------------------------
    breakdowns = {}
    windows = {}
    for t, r in returns_by_asset.items():
        if config.windows and t in config.windows:
            windows[t] = int(config.windows[t])
            continue
        wb = compute_asset_window(t, r, criterion=config.window_criterion, combine=config.window_combine)
        breakdowns[t] = wb
        if wb.w_final is not None:
            windows[t] = wb.w_final
        log(f"  {t}: W_E={wb.w_stabilization:.0f} ({wb.we_status})  W_T={wb.w_testing_power:.0f}  "
            f"W_K={wb.w_kaufman:.0f}  ->  W_i={wb.w_final}")
    returns_by_asset = {t: r for t, r in returns_by_asset.items() if t in windows}

    # --- Pasos 3-4: estimadores -------------------------------------------
    estimators = {
        t: compute_rolling_estimators(
            r, window=windows[t], recalibration_step=config.recalibration_step,
            horizon=config.horizon, cache_dir=config.cache_dir, ticker=t,
            u_multiplier=config.u_multiplier,
        )
        for t, r in returns_by_asset.items()
    }

    # --- Pasos 5-7: señal ---------------------------------------------------
    exposures, fit, panel = generate_exposures(
        estimators, returns_by_asset, horizon=config.horizon, nw_lags=max(1, config.horizon - 1),
        hysteresis_pp=config.hysteresis_pp, walk_forward=config.walk_forward,
        refit_step=config.refit_step, event_definition=config.event_definition,
        tau_method=config.tau_method,
    )
    exposures_signal = exposures
    if config.drawdown_floor:
        exposures = apply_drawdown_floor(exposures, returns_by_asset, **config.drawdown_floor)

    # --- Paso 8: backtest ---------------------------------------------------
    backtest = run_multi_asset_backtest(returns_by_asset, exposures, config.cost_bps_per_side)
    benchmark = equal_weight_benchmark(returns_by_asset, backtest["per_asset"])
    comparison_e3 = (compare(backtest["portfolio_net_return"], benchmark, cash_rate)
                     if len(benchmark) else pd.DataFrame())

    result = {
        "config": config,
        "returns": returns_by_asset,
        "windows": windows,
        "window_breakdowns": breakdowns,
        "estimators": estimators,
        "exposures": exposures,
        "exposures_signal": exposures_signal,
        "fit": fit,
        "panel": panel,
        "backtest": backtest,
        "benchmark_return": benchmark,
        "comparison_e3": comparison_e3,
        "cash_rate": cash_rate,
    }
    result["diagnostics"] = signal_diagnostics(result)
    # --- Paso 8 (Fase 3): cartera con tenencias y comparadores ---------------
    cost_used, cost_label = cost_for_mode(config.cost_mode, price_data, config.spread_window)
    result["cost_used"], result["cost_mode_used"] = cost_used, cost_label
    result["liquidity"] = None
    if len(benchmark):
        result["portfolios"] = run_comparators(result, config.rebalance, merval_log_returns, cost_used)
        if price_data:
            from comparators import _simple_returns_frame, build_comparator_exposures, simulate_portfolio
            E = build_comparator_exposures(result)["Estrategia"]
            port, trades = simulate_portfolio(_simple_returns_frame(result, E.columns), E, cost_used,
                                              config.rebalance, return_trades=True, cash_rate=cash_rate)
            adv = adv_ars({t: price_data[t] for t in E.columns if t in price_data})
            nav_prev = port["nav"].shift(1).fillna(1.0)
            result["liquidity"] = adv_diagnostics(trades[adv.columns], adv, config.aum_ars,
                                                  config.adv_cap_pct, nav_prev=nav_prev)
        result["comparison"] = comparison_table(result["portfolios"], cash_rate)
        result["comparison_by_period"] = metrics_by_period(result["portfolios"], risk_free_rate=cash_rate)
    else:
        result["portfolios"], result["comparison"] = {}, pd.DataFrame()
        result["comparison_by_period"] = pd.DataFrame()
    return result


def _episodes_in(start, end) -> list[str]:
    if start is None or pd.isna(start):
        return []
    out = []
    for name, dates in EPISODE_CATALOG.items():
        if any(start <= pd.Timestamp(d) <= end for d in dates):
            out.append(name)
    return out


def signal_diagnostics(result: dict) -> pd.DataFrame:
    """
    Por activo: ventana, desde cuándo hay señal y exposición, cuántos días y
    episodios del catálogo quedan evaluables, y si la señal "está viva"
    (cuánto varía lambda y cuántas veces cambia la exposición).
    """
    rows = []
    for t, est in result["estimators"].items():
        wb = result["window_breakdowns"].get(t)
        bt = result["backtest"]["per_asset"].get(t, pd.DataFrame())
        exp = result["exposures"][t].dropna()
        has_signal = est[["z_lambda", "z_dlambda"]].notna().all(axis=1)
        inicio_senal = est.index[has_signal][0] if has_signal.any() else pd.NaT
        inicio_exp = exp.index[0] if len(exp) else pd.NaT
        fin = exp.index[-1] if len(exp) else pd.NaT
        lam = est.loc[exp.index, "jump_intensity"] if len(exp) else pd.Series(dtype=float)
        eps = _episodes_in(inicio_exp, fin)
        rows.append({
            "activo": t,
            "W_i": result["windows"][t],
            "estado_W_E": wb.we_status if wb else "fijo",
            "inicio_senal": inicio_senal.date() if pd.notna(inicio_senal) else None,
            "inicio_exposicion": inicio_exp.date() if pd.notna(inicio_exp) else None,
            "dias_evaluables": int(len(bt)),
            "anios_evaluables": round(len(bt) / 252, 2),
            "episodios_cubiertos": len(eps),
            "episodios": ", ".join(eps),
            "cv_lambda_pct": round(100 * lam.std() / lam.mean(), 2) if len(lam) > 1 else np.nan,
            "cambios_exposicion": int((bt["turnover"].iloc[1:] > 0).sum()) if len(bt) else 0,
            "exp_min": round(exp.min(), 1) if len(exp) else np.nan,
            "exp_media": round(exp.mean(), 1) if len(exp) else np.nan,
            "exp_max": round(exp.max(), 1) if len(exp) else np.nan,
        })
    return pd.DataFrame(rows).set_index("activo")


def compare_window_criteria(
    returns_by_asset: dict[str, pd.Series],
    criteria=("parametros", "sin_mu", "estandarizado", "estimadores"),
    combines=("suma", "max"),
    recalibration_step: int = 5,
    **config_kwargs,
) -> pd.DataFrame:
    """
    Corre el pipeline para cada combinación de criterio de W y devuelve solo
    diagnósticos de cobertura y de señal.
    """
    tables = []
    for c in criteria:
        for m in combines:
            cfg = PipelineConfig(window_criterion=c, window_combine=m,
                                 recalibration_step=recalibration_step, **config_kwargs)
            print(f"\n=== {cfg.label()} ===")
            res = run_pipeline(returns_by_asset, cfg)
            d = res["diagnostics"].copy()
            d.insert(0, "criterio_W", c)
            d.insert(1, "combinacion", m)
            tables.append(d.reset_index())
    return pd.concat(tables, ignore_index=True)


def save_results(result: dict, prefix: str = "") -> list[str]:
    """Guarda los resultados de una corrida en CSV."""
    bt = result["backtest"]
    files = []
    f = f"{prefix}portfolio_backtest_result.csv"
    bt["portfolio_net_return"].to_frame("net_return").assign(nav=bt["portfolio_nav"]).to_csv(f); files.append(f)
    for t, df in bt["per_asset"].items():
        f = f"{prefix}backtest_{t.replace('.', '_')}.csv"; df.to_csv(f); files.append(f)
    f = f"{prefix}signal_panel.csv"; result["panel"].to_csv(f, index=False); files.append(f)
    f = f"{prefix}performance_comparison.csv"; result["comparison"].to_csv(f); files.append(f)
    if result.get("liquidity"):
        f = f"{prefix}adv_por_activo.csv"; result["liquidity"]["por_activo"].to_csv(f); files.append(f)
        f = f"{prefix}adv_por_anio.csv"; result["liquidity"]["por_anio"].to_csv(f); files.append(f)
    f = f"{prefix}performance_by_period.csv"; result["comparison_by_period"].to_csv(f, index=False); files.append(f)
    f = f"{prefix}performance_comparison_e3_reference.csv"; result["comparison_e3"].to_csv(f); files.append(f)
    if result["portfolios"]:
        f = f"{prefix}portfolios_nav.csv"
        pd.DataFrame({k: v["nav"] for k, v in result["portfolios"].items()}).to_csv(f); files.append(f)
    f = f"{prefix}signal_diagnostics.csv"; result["diagnostics"].to_csv(f); files.append(f)
    if result["fit"] is not None and result["fit"].history is not None:
        f = f"{prefix}logit_history.csv"; result["fit"].history.to_csv(f); files.append(f)
    f = f"{prefix}cost_table_bps.csv"; cost_table().to_csv(f); files.append(f)
    f = f"{prefix}config.csv"
    pd.Series({k: str(v) for k, v in asdict(result["config"]).items()}).to_csv(f, header=["valor"]); files.append(f)
    return files


def main():
    print(f"Cargando datos de {len(TICKERS)} activos...")
    price_data = load_price_data(tickers=TICKERS)
    returns_by_asset = {t: price_data[t]["log_return"] for t in price_data}

    try:
        merval = load_benchmark("^MERV")["log_return"]
    except Exception as exc:  # el índice es solo referencia: no frena la corrida
        print(f"[WARN] No se pudo cargar el Merval ({exc}); se omite.")
        merval = None

    config = PipelineConfig()
    print(f"\nConfiguración: {config.label()}")
    result = run_pipeline(returns_by_asset, config, merval_log_returns=merval, price_data=price_data)

    print("\n=== Diagnóstico de señal y cobertura ===")
    print(result["diagnostics"].to_string())
    print(f"\n=== Estrategia vs. comparadores (compra inicial y rebalanceo {config.rebalance} con costos) ===")
    print(result["comparison"].round(4).to_string())
    files = save_results(result)
    print("\nGuardado:", ", ".join(files))


if __name__ == "__main__":
    main()
