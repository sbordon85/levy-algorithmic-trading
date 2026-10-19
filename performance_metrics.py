"""
performance_metrics.py

Métricas de "desempeño ajustado por riesgo" de la Hipótesis de la Entrega #1:
Sharpe, Sortino y Calmar en paralelo, más retorno/volatilidad anualizados y
Max Drawdown.

Todas las funciones esperan RETORNOS SIMPLES diarios (no logarítmicos). Desde
la Fase 0 de la Entrega #4, `backtest_engine.py` entrega `net_return` y el
benchmark en retornos simples.

Estado de los parámetros (actualizado Fase 0, Entrega #4):
  - Costo de transacción: CERRADO, con fuente documentada, en
    `backtest_engine.py` (comisión + derecho de mercado + medio spread de
    Corwin-Schultz, por lado).
  - Tasa libre de riesgo (`risk_free_rate`): CERRADA en la Entrega #4 con
    `macro_data.py` (caución a 1 día en pesos; T-bill 3m en la vista en USD).
    Acepta un escalar (TNA anual, se divide por 252) o una SERIE de
    rendimientos DIARIOS alineada por fecha (`macro_data.daily_cash_return`),
    que es lo que usa el pipeline. El default 0.0 se conserva solo para
    compatibilidad: con tasas nominales en pesos altas sobreestima Sharpe y
    Sortino, así que las tablas reportables deben pasar la serie.
    Calmar no se ajusta por tasa (retorno total / max drawdown).

Correcciones de definición (Fase 0, Entrega #4):
  - `max_drawdown` mide la caída desde el capital inicial (NAV = 1) además de
    los picos posteriores. Antes, si la serie arrancaba perdiendo, esa primera
    caída no contaba como drawdown.
  - `sortino_ratio` usa la desviación a la baja estándar
    (Sortino & Price, 1994): sqrt(mean(min(r - rf, 0)^2)) sobre TODAS las
    observaciones. Antes se usaba el desvío muestral de los retornos
    negativos solamente, que no es la definición estándar y resta la media de
    las pérdidas (subestimando la dispersión a la baja).

TODO: definir cuál de las tres métricas es la principal (pregunta abierta de
la Reunión #2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return np.nan
    cumulative = (1 + returns).prod()
    return cumulative ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def _daily_rf(returns: pd.Series, risk_free_rate):
    """Tasa libre de riesgo DIARIA: serie (alineada por fecha), escalar anual o None (= 0)."""
    if risk_free_rate is None:
        return 0.0
    if isinstance(risk_free_rate, pd.Series):
        return risk_free_rate.reindex(returns.index).fillna(0.0)
    return risk_free_rate / TRADING_DAYS_PER_YEAR


def sharpe_ratio(returns: pd.Series, risk_free_rate=0.0) -> float:
    excess = returns - _daily_rf(returns, risk_free_rate)
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS_PER_YEAR)


def downside_deviation(returns: pd.Series, risk_free_rate=0.0) -> float:
    """sqrt(mean(min(r - rf, 0)^2)) sobre todas las observaciones (diaria)."""
    excess = (returns - _daily_rf(returns, risk_free_rate)).dropna()
    if len(excess) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))


def sortino_ratio(returns: pd.Series, risk_free_rate=0.0) -> float:
    excess = returns - _daily_rf(returns, risk_free_rate)
    dd = downside_deviation(returns, risk_free_rate)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return (excess.mean() / dd) * np.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown diario, medido contra el máximo entre el capital inicial (1.0)
    y todos los NAV anteriores."""
    nav = (1 + returns.dropna()).cumprod()
    running_max = nav.cummax().clip(lower=1.0)
    return nav / running_max - 1


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    if dd.empty:
        return np.nan
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(returns) / abs(mdd)


def summary(returns: pd.Series, label: str = "Estrategia", risk_free_rate=0.0) -> pd.Series:
    """Resumen de todas las métricas para una serie de retornos simples.
    `risk_free_rate` (serie diaria o escalar anual) afecta solo a Sharpe y Sortino."""
    return pd.Series(
        {
            "Retorno anualizado": annualized_return(returns),
            "Volatilidad anualizada": annualized_volatility(returns),
            "Sharpe": sharpe_ratio(returns, risk_free_rate),
            "Sortino": sortino_ratio(returns, risk_free_rate),
            "Max Drawdown": max_drawdown(returns),
            "Calmar": calmar_ratio(returns),
            "Días": int(returns.dropna().shape[0]),
        },
        name=label,
    )


def compare(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate=0.0) -> pd.DataFrame:
    """Tabla comparativa estrategia vs. benchmark, alineando fechas comunes."""
    aligned = pd.concat(
        {"Estrategia": strategy_returns, "Benchmark": benchmark_returns}, axis=1
    ).dropna()
    return pd.concat(
        [summary(aligned["Estrategia"], "Estrategia", risk_free_rate),
         summary(aligned["Benchmark"], "Benchmark", risk_free_rate)],
        axis=1,
    )
