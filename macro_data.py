"""
macro_data.py  (Entrega #4)

Variables macro de CONTROL prometidas en la Entrega #2 (sección "Datos de
control/contexto"). No son insumo de las reglas de señal, que dependen solo de
precios; se usan para:

  1. Tasa libre de riesgo en pesos -> Sharpe y Sortino en exceso de la tasa, y
     rendimiento de la CAJA de la cartera (antes la caja rendía 0%, lo que
     favorecía al benchmark 100% invertido).
  2. Tipo de cambio oficial y tasa en dólares -> robustez de los resultados
     expresados en USD (la exposición sigue siendo la del modelo en pesos).

Elecciones (confirmadas con el autor):
  - Tasa en pesos: pase entre terceros a 1 día del BCRA (id_variable 150, TNA, base 365),
    usado como proxy de la caución a 1 día (el BCRA no publica la caución bursátil).
  - Dólares: tipo de cambio OFICIAL del BCRA y T-bill de EE.UU. a 3 meses
    (yfinance ^IRX, tasa de descuento, base 360).

Fuentes y reproducibilidad
--------------------------
Cada serie se lee de un snapshot CSV en `datos/macro/` (columnas `fecha`,
`valor`). Si no existe, se descarga y se guarda el snapshot, para que la
corrida (y la caché del NIG) sea reproducible sin red. Las descargas del BCRA
usan la API pública del BCRA vía `requests`; las descargas de la tasa (v4.0, id 150) y del tipo de cambio oficial (cambiarias v1.0) fueron
verificadas por el autor contra la API real;
`find_bcra_variables("pase")` lista los candidatos (la API v3.0 fue dada de baja; se usa v4.0).
La serie usada es el id 150 ("Tasa de interés por operaciones de pases entre terceros a 1 día").
Si la descarga automática falla, basta con dejar un CSV `fecha,valor` a mano.

Convención temporal (igual que el resto del framework)
------------------------------------------------------
`daily_cash_return` devuelve, para cada rueda t, el rendimiento de tener caja
desde el cierre de t-1 hasta el cierre de t, con la tasa VIGENTE al cierre de
t-1 (sin look-ahead). Los fines de semana y feriados devengan intereses: se
multiplica por los días corridos entre ruedas.

Advertencia sobre el tipo de cambio oficial
-------------------------------------------
Bajo el cepo (oct-2019 a abr-2025) el oficial subestima la depreciación
efectiva del peso (existían el CCL y el MEP). Los resultados en USD con el
oficial deben leerse como una cota, no como el retorno dolarizado realizable.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_DIR = Path("datos/macro")
CAUCION_FILE = "caucion_1d_tna.csv"
FX_FILE = "usd_oficial.csv"
TBILL_FILE = "tbill_3m_tna.csv"

DAY_COUNT_ARS = 365   # caución: TNA sobre 365 días
DAY_COUNT_USD = 360   # T-bill: tasa de descuento sobre 360 días


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------

def _snapshot_path(name: str, directory: str | Path | None = None) -> Path:
    return Path(directory or SNAPSHOT_DIR) / name


def read_snapshot(name: str, directory: str | Path | None = None) -> pd.Series | None:
    """Serie desde CSV (`fecha`,`valor`); None si el archivo no existe o no se puede leer."""
    path = _snapshot_path(name, directory)
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["fecha"])
    except (OSError, ValueError) as exc:
        warnings.warn(f"Snapshot ilegible ({path}): {exc}")
        return None
    s = pd.Series(df["valor"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["fecha"]), name=path.stem)
    return s[~s.index.duplicated(keep="last")].sort_index()


def save_snapshot(series: pd.Series, name: str, directory: str | Path | None = None) -> Path | None:
    """Guarda la serie como CSV. Un fallo de disco avisa pero no corta la corrida."""
    path = _snapshot_path(name, directory)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"fecha": series.index, "valor": series.to_numpy()}).to_csv(path, index=False)
        return path
    except OSError as exc:
        warnings.warn(f"No se pudo guardar el snapshot {path}: {exc}")
        return None


# --------------------------------------------------------------------------
# Descargas (requieren red)
# --------------------------------------------------------------------------

BCRA_BASE = "https://api.bcra.gob.ar"
BCRA_MONETARIAS = "v4.0"   # la v3.0 devuelve 410 Gone (dada de baja)


def _bcra_get(path: str, params: dict | None = None, verify: bool | None = None) -> dict:
    """
    GET a la API pública del BCRA con `requests`. La cadena de certificados del BCRA
    suele fallar en Windows/Python (CERTIFICATE_VERIFY_FAILED): si pasa, se reintenta
    sin verificar el certificado (solo para datos públicos de lectura) y se avisa.
    """
    import requests
    url = BCRA_BASE + path
    try:
        r = requests.get(url, params=params, timeout=60, verify=True if verify is None else verify)
    except requests.exceptions.SSLError:
        if verify is not None:
            raise
        warnings.warn("Falla SSL con api.bcra.gob.ar; se reintenta sin verificar el certificado "
                      "(datos públicos de solo lectura).")
        try:
            import urllib3
            urllib3.disable_warnings()
        except Exception:
            pass
        r = requests.get(url, params=params, timeout=60, verify=False)
    r.raise_for_status()
    return r.json()


def _bcra_rows(payload: dict) -> list:
    """Filas de `results` (la API v4 a veces anida el detalle por variable)."""
    rows = payload.get("results", payload)
    out = []
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("detalle"), list) and "fecha" not in item:
            out.extend(item["detalle"])
        else:
            out.append(item)
    return out


def find_bcra_variables(keyword: str) -> pd.DataFrame:
    """Variables monetarias del BCRA cuya descripción contiene `keyword` (ej. 'cauci', 'BADLAR')."""
    rows, offset, limit = [], 0, 1000
    while True:
        payload = _bcra_get(f"/estadisticas/{BCRA_MONETARIAS}/monetarias", {"limit": limit, "offset": offset})
        chunk = payload.get("results", [])
        rows.extend(chunk)
        total = payload.get("metadata", {}).get("resultset", {}).get("count", len(rows))
        offset += limit
        if not chunk or offset >= total:
            break
    df = pd.DataFrame(rows)
    col = next((c for c in df.columns if "descrip" in c.lower()), None)
    if col is None:
        raise KeyError(f"Columnas devueltas por el BCRA: {list(df.columns)}")
    return df[df[col].astype(str).str.contains(keyword, case=False, na=False)].reset_index(drop=True)


def _bcra_series(path: str, start: str, end: str | None, desde: str, hasta: str) -> pd.DataFrame:
    """Descarga paginada (limit/offset) de una serie del BCRA."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rows, offset, limit = [], 0, 1000
    while True:
        payload = _bcra_get(path, {desde: start, hasta: end, "limit": limit, "offset": offset})
        chunk = _bcra_rows(payload)
        rows.extend(chunk)
        total = payload.get("metadata", {}).get("resultset", {}).get("count", len(rows))
        offset += limit
        if not chunk or offset >= total:
            break
    return pd.DataFrame(rows)


def fetch_bcra_monetary(id_variable, start: str = "2015-01-01", end: str | None = None) -> pd.Series:
    """Serie de una variable monetaria del BCRA (`valor` indexado por `fecha`)."""
    df = _bcra_series(f"/estadisticas/{BCRA_MONETARIAS}/monetarias/{id_variable}", start, end, "desde", "hasta")
    s = pd.Series(df["valor"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(df["fecha"])))
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_bcra_usd_official(start: str = "2015-01-01", end: str | None = None) -> pd.Series:
    """Cotización oficial USD/ARS del BCRA (tipo de cotización de la moneda USD)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    path = "/estadisticascambiarias/v1.0/Cotizaciones/USD"
    rows, offset, limit = [], 0, 1000
    while True:
        payload = _bcra_get(path, {"fechadesde": start, "fechahasta": end, "limit": limit, "offset": offset})
        chunk = payload.get("results", [])
        for item in chunk:
            fecha = item.get("fecha")
            det = item.get("detalle")
            if isinstance(det, list):           # formato: {fecha, detalle: [{codigoMoneda, tipoCotizacion, ...}]}
                for d in det:
                    if d.get("codigoMoneda", "USD") == "USD":
                        rows.append({"fecha": fecha, "valor": d.get("tipoCotizacion")})
            else:                               # formato plano
                rows.append({"fecha": fecha, "valor": item.get("tipoCotizacion")})
        total = payload.get("metadata", {}).get("resultset", {}).get("count", len(rows))
        offset += limit
        if not chunk or offset >= total:
            break
    df = pd.DataFrame(rows)
    if df.empty or df["valor"].isna().all():
        raise ValueError(f"Respuesta sin cotizaciones USD; último payload: {str(payload)[:400]}")
    df = df.dropna(subset=["valor"])
    s = pd.Series(df["valor"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(df["fecha"])))
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_tbill_3m(start: str = "2015-01-01", end: str | None = None) -> pd.Series:
    """T-bill de EE.UU. a 13 semanas (yfinance ^IRX, en % anual)."""
    import yfinance as yf
    raw = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].dropna().astype(float)
    return s.sort_index()


# --------------------------------------------------------------------------
# Cargadores con snapshot
# --------------------------------------------------------------------------

def load_caucion_tna(id_variable=None, start: str = "2015-01-01", end: str | None = None,
                     directory: str | Path | None = None, refresh: bool = False) -> pd.Series:
    """Tasa a 1 día en pesos (proxy de caución: pase entre terceros, id 150), TNA en %. Snapshot -> descarga BCRA (`id_variable`) -> error con instrucciones."""
    s = None if refresh else read_snapshot(CAUCION_FILE, directory)
    if s is None:
        if id_variable is None:
            raise FileNotFoundError(
                f"No hay snapshot de la caución ({_snapshot_path(CAUCION_FILE, directory)}) y no se indicó "
                "id_variable. Usá id_variable=150 (pase entre terceros a 1 día; ver find_bcra_variables('pase')) o dejá un CSV con columnas "
                "`fecha,valor` (TNA en %).")
        s = fetch_bcra_monetary(id_variable, start, end)
        save_snapshot(s, CAUCION_FILE, directory)
    return s


def load_fx_official(start: str = "2015-01-01", end: str | None = None,
                     directory: str | Path | None = None, refresh: bool = False) -> pd.Series:
    """USD/ARS oficial (pesos por dólar)."""
    s = None if refresh else read_snapshot(FX_FILE, directory)
    if s is None:
        s = fetch_bcra_usd_official(start, end)
        save_snapshot(s, FX_FILE, directory)
    return s


def load_tbill_tna(start: str = "2015-01-01", end: str | None = None,
                   directory: str | Path | None = None, refresh: bool = False) -> pd.Series:
    """T-bill 3 meses EE.UU., en % anual."""
    s = None if refresh else read_snapshot(TBILL_FILE, directory)
    if s is None:
        s = fetch_tbill_3m(start, end)
        save_snapshot(s, TBILL_FILE, directory)
    return s


# --------------------------------------------------------------------------
# Transformaciones (puras, testeables sin red)
# --------------------------------------------------------------------------

def daily_cash_return(rate_pct: pd.Series, trading_index: pd.DatetimeIndex,
                      day_count: int = DAY_COUNT_ARS) -> pd.Series:
    """
    Rendimiento SIMPLE de la caja para cada rueda t: desde el cierre de t-1 hasta
    el cierre de t, con la tasa vigente al cierre de t-1 (`rate_pct`, % anual).
    Devenga por días corridos entre ruedas. Antes de la primera observación de
    la tasa el rendimiento es 0 (se avisa cuántas ruedas quedan sin tasa).
    """
    idx = pd.DatetimeIndex(trading_index)
    r = rate_pct.dropna().sort_index()
    on_days = r.reindex(r.index.union(idx)).ffill().reindex(idx)
    known = on_days.shift(1)                      # tasa al cierre de t-1
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(1.0)
    out = (known / 100.0) * days / day_count
    missing = int(out.isna().sum())
    if missing:
        warnings.warn(f"{missing} ruedas sin tasa disponible (antes de la primera observación): caja al 0% ahí.")
    return out.fillna(0.0).rename("cash_rate")


def fx_simple_returns(fx: pd.Series, trading_index: pd.DatetimeIndex) -> pd.Series:
    """Variación simple diaria de USD/ARS en las ruedas de la bolsa (ffill en feriados cambiarios)."""
    idx = pd.DatetimeIndex(trading_index)
    f = fx.dropna().sort_index()
    on_days = f.reindex(f.index.union(idx)).ffill().reindex(idx)
    return on_days.pct_change().rename("fx_return")


def to_usd_returns(simple_returns_ars: pd.DataFrame | pd.Series, fx_return: pd.Series):
    """r_usd = (1 + r_ars) / (1 + variación de USD/ARS) - 1. La primera rueda sin tipo de cambio queda igual (0% de FX)."""
    d = fx_return.reindex(simple_returns_ars.index).fillna(0.0)
    if isinstance(simple_returns_ars, pd.DataFrame):
        return (1.0 + simple_returns_ars).div(1.0 + d, axis=0) - 1.0
    return (1.0 + simple_returns_ars) / (1.0 + d) - 1.0
