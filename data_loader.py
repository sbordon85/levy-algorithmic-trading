"""
data_loader.py

Carga de datos históricos diarios de acciones del panel líder de BYMA vía yfinance.
Fuente validada: yfinance con sufijo ".BA" tiene ~11 años de historia diaria
para GGAL, PAMP, YPFD, BMA (desde 2015) y BYMA (desde 2017, fecha de su listado).

Cortes por activo ("el universo fue depurado de contaminación por iliquidez temprana"): 
GGAL, PAMP, YPFD y BMA se
cortan en 2015-01-01; BYMA se deja SIN corte (limpia desde su salida a
cotizar en 2017, un corte artificial ahí solo recortaría historia útil).
Ver `DEFAULT_START_BY_TICKER`.

Precios ajustados
-------------------------------------------------
Se descarga SIEMPRE con `auto_adjust=False` (explícito) y:
  - `log_return` se calcula sobre `Adj Close` (ajustado por dividendos y
    splits). 
  - `Close`, `High`, `Low` y `Volume` quedan SIN ajustar: son los que
    necesitan el monto negociado (ADV) y el estimador de spread de
    Corwin-Schultz.

Frecuencia: el framework trabaja a frecuencia DIARIA. La mejor frecuencia 
intradiaria disponible en yfinance (1h, ~1085
días) no cubre los episodios 2018, PASO 2019 ni Covid 2020 -- por eso la
frecuencia diaria es una decisión respaldada por evidencia, no solo una
limitación de la fuente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["GGAL.BA", "PAMP.BA", "YPFD.BA", "BMA.BA", "BYMA.BA"]

# Corte por activo. BYMA -> None: sin corte, se usa toda la
# historia que devuelva yfinance (arranca sola en su fecha de listado, 2017).
DEFAULT_START_BY_TICKER: dict[str, str | None] = {
    "GGAL.BA": "2015-01-01",
    "PAMP.BA": "2015-01-01",
    "YPFD.BA": "2015-01-01",
    "BMA.BA": "2015-01-01",
    "BYMA.BA": None,
}

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _download(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Único punto de contacto con yfinance (aislado para poder testearlo)."""
    if start is None:
        # "Sin corte" (BYMA): pasar start=None a yfinance NO trae
        # toda la historia (interpreta la ausencia de fechas como el período
        # por defecto, ~1 mes). Hay que pedir period="max" explícitamente.
        return yf.download(
            ticker, period="max", end=end, progress=False, auto_adjust=False
        )
    return yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False
    )


def process_price_frame(raw: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Limpia un DataFrame crudo de yfinance (auto_adjust=False) y agrega
    `log_return` calculado sobre `Adj Close`.

    - Aplana columnas MultiIndex (yfinance devuelve (campo, ticker)).
    - Descarta filas sin `Adj Close` y precios no positivos.
    - La primera fila queda sin retorno y se elimina.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{ticker}: faltan columnas {missing} en la descarga de yfinance. "
            "¿Se descargó con auto_adjust=True? Este loader requiere auto_adjust=False."
        )
    df = df[PRICE_COLUMNS].copy()
    df = df[df["Adj Close"].notna() & (df["Adj Close"] > 0)]
    df = df.sort_index()
    df["log_return"] = np.log(df["Adj Close"]).diff()
    return df.dropna(subset=["log_return"])


def load_price_data(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str | dict[str, str | None] | None = "__default_by_ticker__",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Descarga precios diarios para una lista de tickers.

    Parameters
    ----------
    start : str | dict | None
        - "__default_by_ticker__" (default): usa `DEFAULT_START_BY_TICKER`,
          el corte por activo (2015-01-01 salvo BYMA, sin corte).
        - str: una única fecha de corte aplicada a TODOS los tickers.
        - dict[ticker] -> fecha o None: corte explícito por ticker.
        - None: sin corte para ningún ticker.

    Returns
    -------
    dict[str, pd.DataFrame]
        Un DataFrame por ticker, indexado por fecha, con columnas
        Open/High/Low/Close/Adj Close/Volume (sin ajustar salvo Adj Close) y
        `log_return` (sobre Adj Close).
    """
    if start == "__default_by_ticker__":
        start_by_ticker = DEFAULT_START_BY_TICKER
    elif isinstance(start, dict):
        start_by_ticker = start
    else:
        start_by_ticker = {t: start for t in tickers}

    data = {}
    for ticker in tickers:
        raw = _download(ticker, start_by_ticker.get(ticker), end)
        if raw is None or raw.empty:
            print(f"[WARN] Sin datos para {ticker}, se omite.")
            continue
        data[ticker] = process_price_frame(raw, ticker)
    return data


def load_benchmark(
    ticker: str = "^MERV",
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Carga el índice S&P Merval con la misma
    convención que `load_price_data` (auto_adjust=False, log_return sobre
    Adj Close). Se usa como referencia de mercado en comparators.py.
    """
    raw = _download(ticker, start, end)
    return process_price_frame(raw, ticker)
