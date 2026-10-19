"""
backtest_engine.py

Aplica la exposición generada por signal_generator.py sobre los retornos
reales del activo, calculando el retorno neto de la estrategia, y arma el
overlay multi-activo equal-weight (Sección 5 de
`cierre_especificacion_estrategia.qmd`).

Convención de retornos (corrección Fase 0, Entrega #4)
------------------------------------------------------
`data_loader.py` entrega retornos LOGARÍTMICOS, que son los correctos para
calibrar el NIG y para acumular retornos a h días en el etiquetado de
eventos. Pero la contabilidad de una cartera se hace con retornos SIMPLES:

    r_simple = exp(r_log) - 1
    retorno bruto de la estrategia = exposición * r_simple
    NAV_t = prod(1 + retorno neto)

En la Entrega #3 se componían los log-retornos como si fueran simples
((1 + r_log).cumprod()), lo que subestima sistemáticamente las subas y
sobreestima las bajas (sesgo de orden r^2/2 por día). Ahora este módulo
recibe log-retornos (misma interfaz que antes) y convierte internamente;
todas las salidas (`net_return`, `nav`, `portfolio_net_return`) son simples,
que es lo que esperan las funciones de `performance_metrics.py`.

Costos de transacción (corrección Fase 0, Entrega #4)
-----------------------------------------------------
El costo se cobra por unidad de turnover, y el turnover es un cambio de
exposición en UNA dirección (comprar o vender una fracción de la
posición). Por lo tanto el costo por unidad de turnover es el costo POR LADO:

    costo por lado = comisión de bróker + derecho de mercado (ambos + IVA)
                     + MEDIO spread (cruzar el libro una vez)

En la Entrega #3 se cobraba el costo round-trip (ida y vuelta) por cada
cambio one-way, es decir, el doble del costo correcto. Un ciclo completo
0% -> 100% -> 0% sigue costando exactamente un round-trip.

Fuentes de los componentes:
  - Comisión de bróker: 0.50% + IVA (referencia IOL, supuesto conservador
    declarado) y derecho de mercado BYMA: 5 pb + IVA, vigente desde
    1-oct-2025 (Comunicado BYMA-COM18777). Ver `costos_comision_derecho_mercado.qmd`.
  - Spread: estimador de Corwin-Schultz (2012), media y percentil 95 por
    activo, calculados sobre datos reales en `restricciones_liquidez.qmd`.
    Acá se usa como constante por activo; el costo variable DIARIO queda
    para la Fase 5 del plan de trabajo.

Control de look-ahead: este motor multiplica la exposición decidida para t
por el retorno realizado en t. Que esa exposición use solo información hasta
t-1 es responsabilidad de levy_estimators.py y signal_generator.py (logit
walk-forward desde la Fase 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Componentes del costo (costos_comision_derecho_mercado.qmd) -----------
IVA = 0.21
BROKER_COMMISSION_BPS = 50.0      # 0.50% por lado, referencia IOL (tramo retail)
MARKET_FEE_BPS = 5.0              # derecho de mercado BYMA, desde 1-oct-2025

# Costo fijo POR LADO, con IVA: (50 + 5) * 1.21 = 66.55 pb
FIXED_COST_BPS_PER_SIDE = (BROKER_COMMISSION_BPS + MARKET_FEE_BPS) * (1 + IVA)
FIXED_COST_BPS_ROUND_TRIP = 2 * FIXED_COST_BPS_PER_SIDE  # 133.1 pb (referencia)

# --- Spread de Corwin-Schultz por activo (restricciones_liquidez.qmd) ------
# Spread COMPLETO estimado (bid-ask). Cada cambio de exposición cruza el libro
# una vez, así que paga la MITAD del spread.
SPREAD_CS_MEAN_BPS: dict[str, float] = {
    "GGAL.BA": 93.5,
    "PAMP.BA": 110.6,
    "YPFD.BA": 91.1,
    "BMA.BA": 122.0,
    "BYMA.BA": 125.0,
}
SPREAD_CS_P95_BPS: dict[str, float] = {
    "GGAL.BA": 357.4,
    "PAMP.BA": 367.6,
    "YPFD.BA": 344.3,
    "BMA.BA": 424.0,
    "BYMA.BA": 400.4,
}


def per_side_costs_from_spread(spread_bps_by_asset: dict[str, float]) -> dict[str, float]:
    """Costo por unidad de turnover one-way = fijo por lado + medio spread."""
    return {
        ticker: round(FIXED_COST_BPS_PER_SIDE + spread / 2.0, 3)
        for ticker, spread in spread_bps_by_asset.items()
    }


# Default del backtest: día promedio (spread medio).
#   GGAL 113.3 | PAMP 121.85 | YPFD 112.1 | BMA 127.55 | BYMA 129.05  (pb por lado)
# = exactamente la mitad de los 226.6-258.1 pb round-trip que la Entrega #3
#   cobraba por cada cambio one-way.
COST_BPS_PER_SIDE_BY_ASSET: dict[str, float] = per_side_costs_from_spread(SPREAD_CS_MEAN_BPS)

# Escenario de estrés (spread p95), para análisis de sensibilidad.
COST_BPS_PER_SIDE_BY_ASSET_P95: dict[str, float] = per_side_costs_from_spread(SPREAD_CS_P95_BPS)

# Fallback para un ticker sin spread estimado: solo el costo fijo por lado.
DEFAULT_COST_BPS_PER_SIDE = FIXED_COST_BPS_PER_SIDE


def cost_table() -> pd.DataFrame:
    """Tabla de costos por activo, en pb, para reportar en la tesina."""
    rows = {}
    for ticker in SPREAD_CS_MEAN_BPS:
        rows[ticker] = {
            "fijo_por_lado": FIXED_COST_BPS_PER_SIDE,
            "medio_spread_medio": SPREAD_CS_MEAN_BPS[ticker] / 2,
            "costo_por_lado_medio": COST_BPS_PER_SIDE_BY_ASSET[ticker],
            "costo_round_trip_medio": 2 * COST_BPS_PER_SIDE_BY_ASSET[ticker],
            "medio_spread_p95": SPREAD_CS_P95_BPS[ticker] / 2,
            "costo_por_lado_p95": COST_BPS_PER_SIDE_BY_ASSET_P95[ticker],
            "costo_round_trip_p95": 2 * COST_BPS_PER_SIDE_BY_ASSET_P95[ticker],
        }
    return pd.DataFrame(rows).T.round(2)


def log_to_simple(log_returns: pd.Series) -> pd.Series:
    """r_simple = exp(r_log) - 1 (expm1 para precisión con retornos chicos)."""
    return np.expm1(log_returns)


BACKTEST_COLUMNS = [
    "log_return", "simple_return", "exposure", "turnover",
    "cost", "gross_return", "net_return", "nav",
]


def run_backtest(
    log_returns: pd.Series,
    exposure: pd.Series,
    cost_bps_per_side: float = DEFAULT_COST_BPS_PER_SIDE,
) -> pd.DataFrame:
    """
    Backtest de UN activo.

    Parameters
    ----------
    log_returns : pd.Series
        Retornos LOGARÍTMICOS diarios del activo (salida de data_loader).
    exposure : pd.Series
        Exposición en fracción [0, 1] decidida con información hasta t-1.
    cost_bps_per_side : float
        Costo en puntos básicos por unidad de turnover ONE-WAY (fijo por
        lado + medio spread). Un ciclo completo 0 -> 1 -> 0 paga
        2 * cost_bps_per_side = costo round-trip.

    Returns
    -------
    pd.DataFrame con columnas BACKTEST_COLUMNS. `net_return` y `nav` están en
    retornos simples. El turnover del primer día es la exposición inicial
    completa (se compra la posición desde caja).
    """
    aligned = pd.concat({"log_return": log_returns, "exposure": exposure}, axis=1).dropna()

    if aligned.empty:
        # Caso limite: activo sin ninguna fecha con exposicion valida.
        # run_multi_asset_backtest lo ignora al promediar (skipna=True).
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    aligned["simple_return"] = log_to_simple(aligned["log_return"])
    aligned["turnover"] = aligned["exposure"].diff().abs().fillna(aligned["exposure"].iloc[0])
    aligned["cost"] = aligned["turnover"] * (cost_bps_per_side / 10_000)
    aligned["gross_return"] = aligned["exposure"] * aligned["simple_return"]
    aligned["net_return"] = aligned["gross_return"] - aligned["cost"]
    aligned["nav"] = (1 + aligned["net_return"]).cumprod()

    return aligned[BACKTEST_COLUMNS]


def run_buy_and_hold(log_returns: pd.Series) -> pd.DataFrame:
    """
    Benchmark pasivo de UN activo: 100% invertido, sin costos (una sola
    compra al inicio, sin rebalanceo). NAV = precio_t / precio_0.

    Nota: el promedio diario de varios `simple_return` de esta función (como
    hace main.py) NO es un buy & hold de cartera -- equivale a una cartera
    equal-weight rebalanceada todos los días, sin costos. Ese punto lo
    señaló el profesor en el feedback de la Entrega #3. Desde la Fase 3 la
    comparación principal usa comparators.simulate_portfolio (tenencias
    explícitas, rebalanceo con costos, compra inicial para todos).
    """
    df = pd.DataFrame({"log_return": log_returns})
    df["simple_return"] = log_to_simple(df["log_return"])
    df["nav"] = (1 + df["simple_return"]).cumprod()
    return df


def run_multi_asset_backtest(
    log_returns_by_asset: dict[str, pd.Series],
    exposure_pct_by_asset: dict[str, pd.Series],
    cost_bps_per_side: float | dict[str, float] = COST_BPS_PER_SIDE_BY_ASSET,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Overlay independiente por activo, combinado equal-weight.

    Cada activo corre su propio backtest (propia exposición, propio W_i,
    propio turnover y costo). El retorno de portafolio en t es el promedio
    simple de los `net_return` (simples) de los activos con dato ese día.
    Promediar retornos simples cada día implica rebalancear a pesos iguales
    todos los días sin costo entre activos. Es la versión de la Entrega #3;
    la cartera con rebalanceo explícito y costos está en comparators.py.

    Parameters
    ----------
    log_returns_by_asset : dict[ticker] -> pd.Series de retornos LOG diarios.
    exposure_pct_by_asset : dict[ticker] -> pd.Series de exposición en
        PUNTOS PORCENTUALES [0, 100] (salida de
        `signal_generator.generate_exposures`).
    cost_bps_per_side : float o dict[ticker] -> float, en pb por unidad de
        turnover one-way. Default: `COST_BPS_PER_SIDE_BY_ASSET` (fijo por
        lado + medio spread medio). Para el escenario de estrés pasar
        `COST_BPS_PER_SIDE_BY_ASSET_P95`.

    Returns
    -------
    dict con:
        "per_asset": dict[ticker] -> pd.DataFrame (salida de run_backtest)
        "portfolio_net_return": pd.Series, retorno simple diario equal-weight
        "portfolio_nav": pd.Series, NAV acumulado del overlay
        "cost_bps_per_side": dict[ticker] -> costo efectivamente usado
    """
    per_asset = {}
    used_costs = {}
    for ticker, log_returns in log_returns_by_asset.items():
        exposure_frac = exposure_pct_by_asset[ticker] / 100.0
        cost = (
            cost_bps_per_side.get(ticker, DEFAULT_COST_BPS_PER_SIDE)
            if isinstance(cost_bps_per_side, dict)
            else cost_bps_per_side
        )
        used_costs[ticker] = cost
        per_asset[ticker] = run_backtest(log_returns, exposure_frac, cost)

    net_returns = pd.concat(
        {ticker: df["net_return"] for ticker, df in per_asset.items()}, axis=1
    )
    portfolio_net_return = net_returns.mean(axis=1, skipna=True)
    portfolio_net_return = portfolio_net_return[net_returns.notna().any(axis=1)]
    portfolio_nav = (1 + portfolio_net_return).cumprod()

    return {
        "per_asset": per_asset,
        "portfolio_net_return": portfolio_net_return,
        "portfolio_nav": portfolio_nav,
        "cost_bps_per_side": used_costs,
    }


def equal_weight_benchmark(
    log_returns_by_asset: dict[str, pd.Series],
    per_asset_backtests: dict[str, pd.DataFrame],
) -> pd.Series:
    """
    Benchmark pasivo comparable a la estrategia: en cada fecha promedia los
    retornos SIMPLES de exactamente los mismos activos que la estrategia tuvo
    activos ese día (no los 5 sin condicionar -- ver nota en main.py).

    Supuesto explícito (feedback de la Entrega #3): promediar retornos
    diarios equivale a rebalancear a pesos iguales TODOS LOS DÍAS, sin
    costo. Se conserva solo como referencia de la Entrega #3; la comparación
    principal está en comparators.py.
    """
    series = {
        t: run_buy_and_hold(log_returns_by_asset[t].loc[df.index])["simple_return"]
        for t, df in per_asset_backtests.items()
        if not df.empty
    }
    if not series:
        # Ningún activo tuvo exposición (p. ej. el logit nunca juntó eventos
        # suficientes): no hay nada que comparar.
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1).mean(axis=1, skipna=True)
