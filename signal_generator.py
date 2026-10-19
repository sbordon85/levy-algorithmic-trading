"""
signal_generator.py

Implementa la funcion de exposicion cerrada en la Seccion 4 de
`cierre_especificacion_estrategia.qmd`:

    P(Evento_{i,t}=1 | z_lambda, z_dlambda) =
        Logit^-1( alpha_i + b1*z_lambda + b2*z_dlambda + b3*z_lambda*z_dlambda )
    Exposicion_{i,t} = 100% * (1 - P_hat_{i,t})

con efectos fijos alpha_i por activo, pendientes comunes al panel (pooled
logit), correccion de rareza de King & Zeng (2001) sobre cada alpha_i,
errores estandar Newey-West de 4 rezagos y banda de histeresis de 5 p.p.

Convencion temporal (vale para todo el framework)
-------------------------------------------------
La fila t de los estimadores usa retornos hasta t-1 (levy_estimators.py). La
exposicion calculada en la fila t se aplica al retorno de t
(backtest_engine.py). Es decir: la decision se toma al cierre de t-1 y rige
desde el retorno de t.

Fase 1 (Entrega #4): entrenamiento point-in-time
------------------------------------------------
Feedback del profesor: "solo pueden entrar eventos cuyo horizonte de cinco
dias ya haya terminado". En la Entrega #3 el logit se ajustaba UNA vez sobre
todo el panel y esos coeficientes se usaban en TODAS las fechas: la
exposicion del dia t dependia de eventos futuros. Ahora (`walk_forward=True`,
default):

  - Cada evento etiquetado en la fecha s tiene una fecha `known_at` = ultima
    rueda de su horizonte. Para decidir la exposicion de t solo entran al
    entrenamiento los eventos con known_at < t (horizonte ya cerrado al
    cierre de t-1).
  - El logit se reajusta cada `refit_step` ruedas (1 = diario, como pide la
    especificacion) con ventana expanding.
  - ybar (tasa de eventos de la muestra de entrenamiento) y tau_i (tasa
    poblacional de King-Zeng) se calculan con informacion hasta t-1.
  - Warm-up explicito: un activo no recibe exposicion hasta tener
    MIN_TRAIN_OBS_PER_ASSET observaciones y MIN_TRAIN_EVENTS_PER_ASSET
    eventos cerrados; el ajuste global exige MIN_TRAIN_EVENTS eventos.
  - Se guarda la historia de coeficientes (`fit.history`).
  - La ausencia de look-ahead se verifico perturbando todos los retornos
    posteriores a una fecha y comprobando que la exposicion hasta esa fecha
    no cambia.

`walk_forward=False` reproduce el ajuste unico de la Entrega #3 (con
look-ahead), solo para cuantificar su efecto. No usar para reportar.

Correccion de alineacion del evento (Fase 1)
--------------------------------------------
Como la exposicion de t se aplica al retorno de t, el evento que corresponde
predecir es el del horizonte que esa decision afecta: r_t + ... + r_{t+h-1}.
La version anterior usaba r_{t+1} + ... + r_{t+h}, salteando el primer dia
afectado. Corregido; known_at = fecha de r_{t+h-1}.

Definicion de evento (Fase 2, requiere decision del director)
-------------------------------------------------------------
- "diaria" (especificacion original): r_{t->t+h} fuera de los percentiles
  5/95 del NIG DIARIO calibrado en t. Comparar un retorno de 5 dias contra
  percentiles de 1 dia hace que el propio modelo implique ~45% de "eventos"
  (ver nig_model.prob_outside), no 10%.
- "horizonte": r_{t->t+h} fuera de los percentiles 5/95 del NIG
  convolucionado a h dias (delta*h, mu*h). Tasa implicada: exactamente 10%.

Tasa poblacional tau_i para King-Zeng:
- "excedencias" (especificacion): frecuencia anual de |z|>2 / 252, con la
  historia del activo hasta t-1. Mide OTRA definicion de evento (retorno
  diario extremo), por eso con la definicion "diaria" produce una
  correccion muy grande de alpha_i.
- "modelo": tasa de eventos implicada por el NIG bajo la MISMA definicion de
  evento usada para etiquetar (0.10 en "horizonte"; promedio de
  p_event_model_diaria hasta t-1 en "diaria").

Errores estandar
----------------
statsmodels (Logit) no expone `get_robustcov_results` en la version usada.
Se implementa a mano el sandwich `Bread * Meat * Bread` con Meat acumulado
DENTRO de cada activo (Newey-West clustered por activo, Bartlett, 4
rezagos); no captura correlacion contemporanea entre activos (Driscoll-Kraay).
Se calcula solo para el ultimo ajuste (`fit.result`), porque la exposicion
usa unicamente los coeficientes puntuales.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from window_selection import EXCEEDANCE_Z

HORIZON_H = 5            # dias habiles, Seccion 4
NW_LAGS = HORIZON_H - 1  # 4 rezagos, Seccion 4
HYSTERESIS_PP = 5.0      # puntos porcentuales, Seccion 4

EVENT_DEFINITIONS = ("diaria", "horizonte")
DEFAULT_EVENT_DEFINITION = "horizonte"    # cerrado en Entrega #4 (antes "diaria", especificacion original)
TAU_METHODS = ("excedencias", "modelo")
DEFAULT_TAU_METHOD = "modelo"             # cerrado en Entrega #4 (antes "excedencias", especificacion original)

# Warm-up del entrenamiento point-in-time (supuestos operativos, no
# especificados en el cierre): un logit con efectos fijos necesita, para
# cada activo, observaciones y eventos suficientes para estimar alpha_i.
MIN_TRAIN_OBS_PER_ASSET = 60
MIN_TRAIN_EVENTS_PER_ASSET = 5
MIN_TRAIN_EVENTS = 20
MIN_HISTORY_FOR_TAU = 252   # ruedas de historia minima para tau por excedencias

FEATURES = ["z_lambda", "z_dlambda", "interaction"]
SLOT_NAMES = {"z_lambda": "z_lambda", "z_dlambda": "z_dlambda", "interaction": "z_lambda:z_dlambda"}


@dataclass
class PooledLogitFit:
    result: "sm.discrete.discrete_model.BinaryResultsWrapper | None"
    alpha_raw: dict[str, float]
    alpha_corrected: dict[str, float]
    slopes: dict[str, float]  # z_lambda, z_dlambda, z_lambda:z_dlambda
    cov_type_used: str
    y_bar: float
    tau: dict[str, float]
    n_obs: int = 0
    n_events: int = 0
    mode: str = "walk_forward"
    history: pd.DataFrame | None = field(default=None, repr=False)
    bse_hac: pd.Series | None = field(default=None, repr=False)


# --------------------------------------------------------------------------
# Etiquetado de eventos y panel
# --------------------------------------------------------------------------

def label_adverse_events(
    returns: pd.Series,
    tail_low: pd.Series,
    tail_high: pd.Series,
    horizon: int = HORIZON_H,
) -> tuple[pd.Series, pd.Series]:
    """
    Evento_t = 1 si r_t + ... + r_{t+h-1} (log) cae fuera de [tail_low_t,
    tail_high_t], donde los umbrales de la fila t se calibran con datos hasta
    t-1. Devuelve (event, known_at): known_at es la fecha de r_{t+h-1}, a
    partir de cuyo cierre el evento es observable. Sin horizonte completo
    dentro de la muestra, event = NaN (nunca se imputa).
    """
    fwd_cum = returns.rolling(horizon).sum().shift(-(horizon - 1))
    known_at = pd.Series(returns.index, index=returns.index).shift(-(horizon - 1))
    event = pd.Series(np.nan, index=returns.index)
    valid = fwd_cum.notna() & tail_low.notna() & tail_high.notna()
    event.loc[valid] = (
        (fwd_cum[valid] < tail_low[valid]) | (fwd_cum[valid] > tail_high[valid])
    ).astype(float)
    return event, known_at


def build_panel(
    estimators_by_asset: dict[str, pd.DataFrame],
    returns_by_asset: dict[str, pd.Series],
    horizon: int = HORIZON_H,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
) -> pd.DataFrame:
    """Panel largo (una fila por activo-fecha) para el logit pooled."""
    if event_definition not in EVENT_DEFINITIONS:
        raise ValueError(f"event_definition desconocida: {event_definition}. Opciones: {EVENT_DEFINITIONS}")
    lo_col, hi_col = ("tail_low", "tail_high") if event_definition == "diaria" else ("tail_low_h", "tail_high_h")
    frames = []
    for ticker, est in estimators_by_asset.items():
        rets = returns_by_asset[ticker].reindex(est.index)
        event, known_at = label_adverse_events(rets, est[lo_col], est[hi_col], horizon)
        p_model = est["p_event_model_diaria"] if event_definition == "diaria" else pd.Series(0.10, index=est.index)
        p_model = p_model.where(est[lo_col].notna())
        frames.append(pd.DataFrame({
            "asset": ticker,
            "date": est.index,
            "z_lambda": est["z_lambda"].to_numpy(),
            "z_dlambda": est["z_dlambda"].to_numpy(),
            "event": event.to_numpy(),
            "known_at": known_at.to_numpy(),
            "p_event_model": p_model.to_numpy(),
        }))
    panel = pd.concat(frames, ignore_index=True)
    panel["interaction"] = panel["z_lambda"] * panel["z_dlambda"]
    return panel


# --------------------------------------------------------------------------
# Tasa poblacional tau_i (King-Zeng), point-in-time
# --------------------------------------------------------------------------

def _tau_excedencias(returns: pd.Series, before: pd.Timestamp) -> float:
    """f_bar/252 con la historia del activo estrictamente anterior a `before`."""
    x = returns[returns.index < before].dropna().to_numpy()
    if len(x) < MIN_HISTORY_FOR_TAU:
        return np.nan
    sd = x.std()
    if sd <= 0:
        return np.nan
    z = (x - x.mean()) / sd
    freq_anual = (np.abs(z) > EXCEEDANCE_Z).sum() / (len(x) / 252.0)
    return float(freq_anual / 252.0)


def _tau_modelo(panel_asset: pd.DataFrame, before: pd.Timestamp) -> float:
    vals = panel_asset.loc[panel_asset["date"] < before, "p_event_model"].dropna()
    return float(vals.mean()) if len(vals) else np.nan


def king_zeng_correction(alpha_hat: float, tau: float, y_bar: float) -> float:
    """alpha_corr = alpha_hat - ln( (1-tau)/tau * ybar/(1-ybar) )."""
    if tau is None or not np.isfinite(tau) or not (0 < tau < 1) or not (0 < y_bar < 1):
        return alpha_hat
    return alpha_hat - np.log((1 - tau) / tau * y_bar / (1 - y_bar))


# --------------------------------------------------------------------------
# Ajuste del logit pooled
# --------------------------------------------------------------------------

def _design(train: pd.DataFrame, assets: list[str]) -> np.ndarray:
    dummies = np.column_stack([(train["asset"].to_numpy() == a).astype(float) for a in assets])
    return np.column_stack([dummies, train[FEATURES].to_numpy(dtype=float)])


def _fit_logit(train: pd.DataFrame, assets: list[str], start_params=None):
    X = _design(train, assets)
    y = train["event"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y, X)
        try:
            res = model.fit(disp=False, start_params=start_params, maxiter=100)
        except Exception:
            return None
    if not np.all(np.isfinite(res.params)):
        return None
    return res


def _eligible_assets(train: pd.DataFrame) -> list[str]:
    g = train.groupby("asset")["event"].agg(["size", "sum"])
    ok = g[(g["size"] >= MIN_TRAIN_OBS_PER_ASSET)
           & (g["sum"] >= MIN_TRAIN_EVENTS_PER_ASSET)
           & (g["size"] - g["sum"] >= MIN_TRAIN_EVENTS_PER_ASSET)]
    return sorted(ok.index.tolist())


def _package_fit(res, assets, train, tau_by_asset, mode) -> PooledLogitFit:
    params = res.params
    k = len(assets)
    alpha_raw = {a: float(params[i]) for i, a in enumerate(assets)}
    slopes = {SLOT_NAMES[f]: float(params[k + j]) for j, f in enumerate(FEATURES)}
    y_bar = float(train["event"].mean())
    alpha_corr = {a: king_zeng_correction(alpha_raw[a], tau_by_asset.get(a, np.nan), y_bar) for a in assets}
    return PooledLogitFit(
        result=res, alpha_raw=alpha_raw, alpha_corrected=alpha_corr, slopes=slopes,
        cov_type_used="nonrobust (HAC solo en el ultimo ajuste)", y_bar=y_bar,
        tau={a: tau_by_asset.get(a, np.nan) for a in assets},
        n_obs=int(len(train)), n_events=int(train["event"].sum()), mode=mode,
    )


def _attach_hac(fit: PooledLogitFit, train: pd.DataFrame, assets: list[str], nw_lags: int) -> None:
    order = train.sort_values(["asset", "date"])
    res = _fit_logit(order, assets, start_params=fit.result.params)
    if res is None:
        return
    groups = order["asset"].to_numpy()
    time = order.groupby("asset").cumcount().to_numpy()
    cov = _newey_west_panel_cov(res, groups, time, nw_lags)
    names = [f"alpha[{a}]" for a in assets] + [SLOT_NAMES[f] for f in FEATURES]
    if cov is not None:
        fit.bse_hac = pd.Series(np.sqrt(np.diag(cov)), index=names)
        fit.cov_type_used = "Newey-West clustered-por-activo (manual), ultimo ajuste"
    else:
        fit.bse_hac = pd.Series(np.asarray(res.bse), index=names)
        fit.cov_type_used = "nonrobust (el sandwich HAC no pudo calcularse)"


def _newey_west_panel_cov(result, groups: np.ndarray, time: np.ndarray, maxlags: int):
    """Sandwich Newey-West clustered por activo (ver docstring del modulo)."""
    try:
        score = result.model.score_obs(result.params)
        bread = np.asarray(result.cov_params())
    except Exception:
        return None
    k = score.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        s = score[idx[np.argsort(time[idx])]]
        m = len(s)
        meat += s.T @ s
        for lag in range(1, maxlags + 1):
            if m <= lag:
                continue
            w = 1.0 - lag / (maxlags + 1)
            contrib = s[lag:].T @ s[:-lag]
            meat += w * (contrib + contrib.T)
    cov = bread @ meat @ bread
    return cov if np.isfinite(cov).all() else None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _predict(rows: pd.DataFrame, fit: PooledLogitFit) -> np.ndarray:
    alpha = rows["asset"].map(fit.alpha_corrected).to_numpy(dtype=float)
    lin = (alpha
           + fit.slopes["z_lambda"] * rows["z_lambda"].to_numpy(dtype=float)
           + fit.slopes["z_dlambda"] * rows["z_dlambda"].to_numpy(dtype=float)
           + fit.slopes["z_lambda:z_dlambda"] * rows["interaction"].to_numpy(dtype=float))
    return _sigmoid(lin)


# --------------------------------------------------------------------------
# Histeresis
# --------------------------------------------------------------------------

def apply_hysteresis(
    exposure_raw: pd.Series,
    band_pp: float = HYSTERESIS_PP,
    initial_exposure: float | None = None,
) -> pd.Series:
    """
    Banda de histeresis (Seccion 4): se mantiene el valor vigente si
    |Delta Exposicion_t| < band_pp respecto del ultimo valor aplicado. Se
    procesa en orden temporal, para UN activo.
    """
    out = pd.Series(np.nan, index=exposure_raw.index)
    current = initial_exposure
    for dt, raw in exposure_raw.items():
        if pd.isna(raw):
            out.loc[dt] = current
            continue
        if current is None or abs(raw - current) >= band_pp:
            current = raw
        out.loc[dt] = current
    return out


# --------------------------------------------------------------------------
# Pipeline de la Seccion 4
# --------------------------------------------------------------------------

def _tau_at(method, returns_by_asset, panel, assets, before) -> dict[str, float]:
    if method == "excedencias":
        return {a: _tau_excedencias(returns_by_asset[a], before) for a in assets}
    return {a: _tau_modelo(panel[panel["asset"] == a], before) for a in assets}


def generate_exposures(
    estimators_by_asset: dict[str, pd.DataFrame],
    returns_by_asset: dict[str, pd.Series],
    horizon: int = HORIZON_H,
    nw_lags: int = NW_LAGS,
    hysteresis_pp: float = HYSTERESIS_PP,
    walk_forward: bool = True,
    refit_step: int = 1,
    event_definition: str = DEFAULT_EVENT_DEFINITION,
    tau_method: str = DEFAULT_TAU_METHOD,
) -> tuple[dict[str, pd.Series], PooledLogitFit | None, pd.DataFrame]:
    """
    Etiqueta eventos, ajusta el logit pooled (point-in-time si
    walk_forward=True), calcula la exposicion cruda y aplica la histeresis.

    Returns
    -------
    exposures : dict[ticker] -> exposicion en [0, 100] con histeresis,
        indexada como returns_by_asset[ticker] (NaN antes del warm-up).
    fit : ultimo ajuste (con errores HAC y `history` de todos los ajustes),
        o None si nunca hubo datos suficientes.
    panel : panel largo con p_hat, exposure_raw, fecha del ajuste usado
        (`fit_date`) y known_at, para auditoria.
    """
    if tau_method not in TAU_METHODS:
        raise ValueError(f"tau_method desconocido: {tau_method}. Opciones: {TAU_METHODS}")
    panel = build_panel(estimators_by_asset, returns_by_asset, horizon, event_definition)
    panel["p_hat"] = np.nan
    panel["exposure_raw"] = np.nan
    panel["fit_date"] = pd.NaT

    has_x = panel[FEATURES].notna().all(axis=1)
    labeled = has_x & panel["event"].notna()

    if walk_forward:
        fit, history = _walk_forward(panel, has_x, labeled, returns_by_asset, refit_step, tau_method)
        if fit is not None:
            # Errores HAC sobre la MISMA muestra del ultimo ajuste.
            last_t = history.index[-1]
            train = panel[labeled & (panel["known_at"] < last_t)]
            train = train[train["asset"].isin(list(fit.alpha_raw))]
            _attach_hac(fit, train, sorted(fit.alpha_raw), nw_lags)
            fit.history = history
    else:
        fit = _in_sample(panel, has_x, labeled, returns_by_asset, tau_method, nw_lags)

    exposures = {}
    for ticker in estimators_by_asset:
        sub = panel.loc[panel["asset"] == ticker].set_index("date")["exposure_raw"]
        sub = sub.reindex(returns_by_asset[ticker].index)
        exposures[ticker] = apply_hysteresis(sub, band_pp=hysteresis_pp)

    return exposures, fit, panel


def _walk_forward(panel, has_x, labeled, returns_by_asset, refit_step, tau_method):
    dates = np.sort(panel.loc[has_x, "date"].unique())
    lab = panel[labeled]
    current = None
    current_assets = None
    history = []
    for k, t in enumerate(dates):
        t = pd.Timestamp(t)
        if current is None or k % refit_step == 0:
            train = lab[lab["known_at"] < t]
            assets = _eligible_assets(train) if len(train) else []
            train = train[train["asset"].isin(assets)]
            if assets and train["event"].sum() >= MIN_TRAIN_EVENTS:
                start = None
                if current is not None and current_assets == assets:
                    start = current.result.params
                res = _fit_logit(train, assets, start_params=start)
                if res is not None:
                    tau = _tau_at(tau_method, returns_by_asset, panel, assets, t)
                    current = _package_fit(res, assets, train, tau, "walk_forward")
                    current_assets = assets
                    history.append({
                        "fit_date": t, "n_obs": current.n_obs, "n_events": current.n_events,
                        "y_bar": current.y_bar, "assets": ",".join(assets),
                        **{f"alpha_raw[{a}]": v for a, v in current.alpha_raw.items()},
                        **{f"alpha_corr[{a}]": v for a, v in current.alpha_corrected.items()},
                        **{f"tau[{a}]": v for a, v in current.tau.items()},
                        **current.slopes,
                    })
        if current is None:
            continue
        rows = panel.index[has_x & (panel["date"] == t) & panel["asset"].isin(current_assets)]
        if len(rows):
            p = _predict(panel.loc[rows], current)
            panel.loc[rows, "p_hat"] = p
            panel.loc[rows, "exposure_raw"] = 100.0 * (1.0 - p)
            panel.loc[rows, "fit_date"] = history[-1]["fit_date"]
    hist = pd.DataFrame(history).set_index("fit_date") if history else pd.DataFrame()
    return current, hist


def _in_sample(panel, has_x, labeled, returns_by_asset, tau_method, nw_lags):
    """Ajuste unico sobre todo el panel (Entrega #3). CON look-ahead."""
    train = panel[labeled]
    assets = _eligible_assets(train)
    train = train[train["asset"].isin(assets)]
    res = _fit_logit(train, assets) if assets else None
    if res is None:
        return None
    after_all = panel["date"].max() + pd.Timedelta(days=1)
    tau = _tau_at(tau_method, returns_by_asset, panel, assets, after_all)
    fit = _package_fit(res, assets, train, tau, "in_sample (con look-ahead)")
    _attach_hac(fit, train, assets, nw_lags)
    rows = panel.index[has_x & panel["asset"].isin(assets)]
    p = _predict(panel.loc[rows], fit)
    panel.loc[rows, "p_hat"] = p
    panel.loc[rows, "exposure_raw"] = 100.0 * (1.0 - p)
    return fit
