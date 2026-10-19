"""
levy_estimators.py

    Estimador                  Definición cerrada
    --------------------------  ----------------------------------------------
    Intensidad de saltos        integral_{|x|>u} nu(x) dx, medida de Levy del
                                 NIG, u_t = 2*sigma_hat_t
    Sesgo de asimetría           parámetro beta del NIG (directo)
    Probabilidad de cola         percentiles 5/95 via CDF cerrada del NIG
    Aceleración                  lambda_t - lambda_{t-5}, estandarizada
                                 (z_lambda, z_dlambda) dentro de la ventana W_i

Control de look-ahead: la calibración de la fila t usa exclusivamente
`returns.iloc[t-W:t]`, nunca el propio retorno de t. Todo lo que se calcula
en la fila t (parámetros, lambda, percentiles, z-scores) es conocido al
cierre de t-1 y se usa para decidir la exposición aplicada al retorno de t.
Se verifico perturbando los retornos futuros.

Costo computacional: la especificación pide reestimación diaria.
`recalibration_step` permite recalibrar cada N ruedas (propagando el último
ajuste válido). Las cantidades derivadas (lambda, percentiles) solo se
recalculan cuando cambian los parámetros.

-------------------------------
- `tail_low_h`, `tail_high_h`: percentiles 5/95 del retorno acumulado a
  h días (NIG convolucionado: delta*h, mu*h). Son los que corresponde usar
  para etiquetar un evento sobre r_{t->t+h} (variante "horizonte").
- `p_event_model_diaria`: probabilidad, implicada por el propio NIG, de que
  r_{t->t+h} caiga fuera de los percentiles diarios 5/95 (definición de
  evento de la especificación original).
- Cache en disco (`cache_dir`): la calibración se guarda por (ticker, W,
  recalibration_step, huella de los datos).
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from nig_model import (
    fit_nig,
    jump_intensity,
    prob_outside,
    tail_quantiles,
    tail_quantiles_horizon,
)

ACCELERATION_LAG = 5  # días hábiles
DEFAULT_U_MULTIPLIER = 2.0  # u = 2 * sigma_hat
Z_SCORE_MIN_PERIODS = 30  
DEFAULT_HORIZON = 5       

_RAW_COLUMNS = [
    "alpha", "beta", "delta", "mu", "sigma",
    "jump_intensity", "tail_low", "tail_high",
    "tail_low_h", "tail_high_h", "p_event_model_diaria",
]


def _data_fingerprint(returns: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(returns.to_numpy(), dtype=float).tobytes())
    h.update(str(returns.index[0]).encode())
    h.update(str(returns.index[-1]).encode())
    return h.hexdigest()[:16]


def _cache_path(cache_dir, ticker, window, step, horizon, returns) -> Path:
    name = f"{ticker or 'serie'}_W{window}_step{step}_h{horizon}_{_data_fingerprint(returns)}.pkl"
    return Path(cache_dir) / name


def _derived(params, horizon: int, u_multiplier: float) -> dict:
    sigma = params.sigma
    lo, hi = tail_quantiles(params)
    lo_h, hi_h = tail_quantiles_horizon(params, horizon)
    return {
        "sigma": sigma,
        "jump_intensity": jump_intensity(params, u_multiplier * sigma),
        "tail_low": lo,
        "tail_high": hi,
        "tail_low_h": lo_h,
        "tail_high_h": hi_h,
        "p_event_model_diaria": prob_outside(params, lo, hi, horizon),
    }


def _rederive(raw: pd.DataFrame, horizon: int, u_multiplier: float) -> pd.DataFrame:
    """
    Recalcula las cantidades derivadas (lambda con otro u, percentiles a otro
    h) a partir de los parámetros NIG ya calibrados, sin volver a hacer el
    maximum likelihood estimation. Se usa en el análisis de sensibilidad: u y h no cambian la
    calibración.
    """
    from nig_model import NIGParams
    out = raw.copy()
    keys = raw[["alpha", "beta", "delta", "mu"]]
    valid = keys.notna().all(axis=1)
    cache = {}
    for i in np.flatnonzero(valid.to_numpy()):
        k = tuple(keys.iloc[i])
        if k not in cache:
            cache[k] = _derived(NIGParams(*k), horizon, u_multiplier)
        for c, v in cache[k].items():
            out.iat[i, out.columns.get_loc(c)] = v
    return out


def _calibrate(returns: pd.Series, window: int, recalibration_step: int, horizon: int,
               u_multiplier: float = DEFAULT_U_MULTIPLIER) -> pd.DataFrame:
    idx = returns.index
    n = len(returns)
    records = {c: np.full(n, np.nan) for c in _RAW_COLUMNS}

    values = returns.to_numpy()
    last_params = None
    derived = None
    for i in range(window, n):
        if last_params is None or (i - window) % recalibration_step == 0:
            # Un ajuste fallido deja NaN (no se rellena con el ajuste anterior).
            last_params = fit_nig(values[i - window : i])
            derived = None
        params = last_params
        if params is None:
            continue
        sigma = params.sigma
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        if derived is None:
            derived = _derived(params, horizon, u_multiplier)
        records["alpha"][i] = params.alpha
        records["beta"][i] = params.beta
        records["delta"][i] = params.delta
        records["mu"][i] = params.mu
        for k, v in derived.items():
            records[k][i] = v

    return pd.DataFrame(records, index=idx)


def compute_rolling_estimators(
    returns: pd.Series,
    window: int,
    recalibration_step: int = 1,
    horizon: int = DEFAULT_HORIZON,
    cache_dir: str | Path | None = None,
    ticker: str | None = None,
    u_multiplier: float = DEFAULT_U_MULTIPLIER,
) -> pd.DataFrame:
    """
    Calcula, walk-forward y sin look-ahead, los estimadores calibrados del
    NIG sobre una ventana rodante de longitud fija `window` (= W_i).

    Returns
    -------
    pd.DataFrame indexado igual que `returns`, con columnas:
        alpha, beta, delta, mu, sigma  (parámetros NIG, por diagnóstico)
        jump_intensity                 lambda_hat_t
        asymmetry                      beta_t (alias de beta)
        tail_low, tail_high            percentiles 5/95 del NIG diarios en t
        tail_low_h, tail_high_h        percentiles 5/95 del NIG a h días en t
        p_event_model_diaria           P(r_{t->t+h} fuera de tail_low/high)
        intensity_change               lambda_t - lambda_{t-5}
        z_lambda, z_dlambda            estandarización dentro de la ventana W
    """
    # La cache guarda la calibración con el h y u de la especificación (5 y
    # 2*sigma).
    raw = None
    path = None
    if cache_dir is not None:
        path = _cache_path(cache_dir, ticker, window, recalibration_step, DEFAULT_HORIZON, returns)
        try:
            if path.exists():
                raw = pd.read_pickle(path)
        except (OSError, EOFError, ValueError) as exc:
            # La cache es solo una optimización: si el disco no responde o el
            # archivo esta corrupto, se recalcula en vez de cortar la corrida.
            warnings.warn(f"Cache ilegible ({path}): {exc}. Se recalcula.")
            raw = None
    if raw is None:
        raw = _calibrate(returns, window, recalibration_step, DEFAULT_HORIZON, DEFAULT_U_MULTIPLIER)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                raw.to_pickle(path)
            except OSError as exc:
                # No perder minutos de calibración por un fallo al guardar.
                warnings.warn(f"No se pudo guardar la cache en {path.parent}: {exc}. "
                              "Se continua sin cache (revisar que la unidad este disponible).")
    if horizon != DEFAULT_HORIZON or u_multiplier != DEFAULT_U_MULTIPLIER:
        raw = _rederive(raw, horizon, u_multiplier)

    df = raw.copy()
    df["asymmetry"] = df["beta"]
    df["intensity_change"] = df["jump_intensity"] - df["jump_intensity"].shift(ACCELERATION_LAG)

    # Estandarización dentro de la ventana W_i, con un piso fijo
    # de observaciones (Z_SCORE_MIN_PERIODS). Un piso proporcional a W_i
    # vaciaba el panel cuando W_i quedaba cerca del largo de la serie.
    min_periods = Z_SCORE_MIN_PERIODS
    roll_lambda = df["jump_intensity"].rolling(window, min_periods=min_periods)
    df["z_lambda"] = (df["jump_intensity"] - roll_lambda.mean()) / roll_lambda.std(ddof=0)

    roll_dlambda = df["intensity_change"].rolling(window, min_periods=min_periods)
    df["z_dlambda"] = (df["intensity_change"] - roll_dlambda.mean()) / roll_dlambda.std(ddof=0)

    ordered_cols = [
        "alpha", "beta", "delta", "mu", "sigma",
        "jump_intensity", "asymmetry", "tail_low", "tail_high",
        "tail_low_h", "tail_high_h", "p_event_model_diaria",
        "intensity_change", "z_lambda", "z_dlambda",
    ]
    return df[ordered_cols]
