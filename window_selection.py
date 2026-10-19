"""
window_selection.py

Implementa la fórmula de cierre para la ventana rodante W_i:

    W_i = max( W_E,i + W_T,i ,  W_i^Kaufman )

Cada componente es una función determinista de los datos de un activo -
no hay parámetro a elegir "a mano" salvo la grilla de búsqueda, que la
especificación fija en {250, ..., 2500} pero sin indicar el paso. Se usa
un paso de 50 días hábiles (documentado en GRID_STEP) como discretización
práctica de esa grilla; cualquier paso mas fino cambia el resultado en, a
lo sumo, GRID_STEP días.

Supuesto operativo explicito para W_i^Kaufman ("piso cualitativo"):
la especificación pide la "menor W candidata que cubre >=4 de los 5
episodios del catalogo", sin fijar el criterio de detección. Se adopta el
criterio mas directamente verificable con los datos ya disponibles en este
framework: un episodio queda "cubierto" por la ventana W si, calculando la
intensidad de saltos empírica (proxy de excedencias |z|>2) 
sobre una ventana rodante de longitud W, el z-score de esa intensidad 
supera EPISODE_DETECTION_Z en alguna fecha dentro de una tolerancia de 
+/-EPISODE_DATE_TOLERANCE_DAYS ruedas alrededor de alguna de las fechas 
listadas para ese episodio. Este supuesto queda aislado en `_episode_is_covered` 
para poder ajustarlo sin tocar el resto del modulo solo si se decide cambiar el criterio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nig_model import fit_nig, jump_intensity, tail_quantiles, standard_errors

# --- Grilla y umbrales (documentados arriba / en cada componente) ---------
GRID_MIN = 250
GRID_MAX = 2500
GRID_STEP = 50
STABILIZATION_REL_CHANGE = 0.10  # 10%
STANDARDIZED_MAX_CHANGE = 1.0    # criterio "estandarizado": cambio < 1 error estándar

# --- Criterios de estabilización de W_E -------------
# "parámetros"    : especificación original. Cambio relativo de
#                   (alpha, beta, delta, mu) < 10% entre pasos de la grilla.
#                   Problema: mu ~ 2e-4, asi que cualquier variación mínima es
#                   un cambio relativo enorme -> W_E se va al techo de la grilla
#                   (4 de 5 activos en las corridas reales).
# "sin_mu"        : igual que el anterior pero sin mu. Advertencia: beta
#                   también puede estar cerca de cero en algún activo, con el
#                   mismo problema.
# "estandarizado" : cambio de cada parámetro (incluido mu) medido en unidades
#                   de su propio error estándar (Hessiano del maximum likelihood estimation):
#                   max_k |theta_k(N') - theta_k(N)| / SE_k(N') < 1.
#                   No depende de la escala del parámetro.
# "estimadores"   : estabilidad de lo que la estrategia efectivamente usa:
#                   cambio relativo de (lambda_hat, percentil 5, percentil 95,
#                   sigma) < 10%. Ninguna de esas cantidades esta cerca de cero.
WINDOW_CRITERIA = ("parametros", "sin_mu", "estandarizado", "estimadores")
DEFAULT_WINDOW_CRITERION = "estimadores"  

# Como se combinan W_E y W_T:
# "suma" : especificación original, W_i = max(W_E + W_T, W_Kaufman).
# "max"  : W_i = max(W_E, W_T, W_Kaufman). Justificación: W_E es la historia
#          necesaria para que el NIG se estabilice y W_T la necesaria para
#          acumular ~10 excedencias; son dos requisitos sobre la misma ventana
#          que se cumplen en paralelo, no uno después del otro.
WINDOW_COMBINE = ("suma", "max")
DEFAULT_WINDOW_COMBINE = "max"          # elegido con la corrida 1; "suma" = especificación original
EXCEEDANCE_Z = 2.0               # |z|>2
EPISODE_DETECTION_Z = 1.0        # supuesto operativo
EPISODE_DATE_TOLERANCE_DAYS = 5  # ruedas de tolerancia alrededor de cada fecha de episodio
BYMA_RELAXED_EVENT_TARGET = 6    # excepcion BYMA: objetivo relajado a 6 eventos
DEFAULT_EVENT_TARGET = 4         # objetivo estándar: cubrir >=4 de 5 episodios

# Catalogo de episodios sistémicos verificados
EPISODE_CATALOG: dict[str, list[str]] = {
    "crisis_cambiaria_2018": ["2018-06-18", "2018-08-31"],
    "paso_2019": ["2019-08-12", "2019-08-20", "2019-09-03"],
    "covid_petroleo_2020": ["2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18", "2020-04-28"],
    "ciclo_electoral_2023": ["2023-10-17", "2023-10-23", "2023-11-21"],
    "era_milei_2024_2025": [
        "2024-07-15", "2025-04-09", "2025-09-08", "2025-09-22", "2025-10-27",
    ],
}


def _grid() -> np.ndarray:
    return np.arange(GRID_MIN, GRID_MAX + 1, GRID_STEP)


@dataclass(frozen=True)
class WindowBreakdown:
    ticker: str
    w_stabilization: float  # W_E,i
    w_testing_power: float  # W_T,i
    w_kaufman: float        # W_i^Kaufman
    w_final: int             # W_i
    byma_exception_applied: bool = False
    notes: list[str] = field(default_factory=list)
    criterion: str = DEFAULT_WINDOW_CRITERION
    combine: str = DEFAULT_WINDOW_COMBINE
    # "estabiliza"     : hay un N de la grilla desde el cual el criterio se
    #                    cumple en todos los N' mayores, con al menos 2 pasos
    #                    posteriores de evidencia.
    # "en_borde"       : el criterio solo se cumple en el último punto (o
    #                    penúltimo) de la grilla.
    # "no_estabiliza"  : nunca se cumple; W_E = techo de la grilla.
    we_status: str = ""


def _stability_vector(x: np.ndarray, criterion: str):
    """Vector de cantidades cuya estabilidad se evalúa, y su escala."""
    params = fit_nig(x)
    if params is None:
        return None
    theta = np.array([params.alpha, params.beta, params.delta, params.mu])
    if criterion == "parámetros":
        return theta, None
    if criterion == "sin_mu":
        return theta[:3], None
    if criterion == "estandarizado":
        return theta, standard_errors(x, params)
    if criterion == "estimadores":
        sigma = params.sigma
        lam = jump_intensity(params, 2.0 * sigma)
        lo, hi = tail_quantiles(params)
        return np.array([lam, lo, hi, sigma]), None
    raise ValueError(f"criterio desconocido: {criterion}. Opciones: {WINDOW_CRITERIA}")


def stabilization_window(
    returns: pd.Series,
    grid: np.ndarray | None = None,
    criterion: str = DEFAULT_WINDOW_CRITERION,
    return_status: bool = False,
):
    """
    W_E,i: mínimo N de la grilla desde el cual la medida de cambio entre
    pasos consecutivos se mantiene por debajo del umbral en TODOS los N'
    mayores. Se calibra siempre con los últimos N retornos
    disponibles del activo. `criterion` elige que se mide (ver
    WINDOW_CRITERIA arriba).

    Si `return_status=True` devuelve (W_E, estado) con estado en
    {"estabiliza", "en_borde", "no_estabiliza"}.
    """
    grid = grid if grid is not None else _grid()
    x = returns.dropna().to_numpy()
    vecs = {}
    for n in grid:
        if n > len(x):
            continue
        out = _stability_vector(x[-n:], criterion)
        if out is not None:
            vecs[int(n)] = out

    ns = sorted(vecs.keys())
    if len(ns) < 2:
        w = float(ns[0]) if ns else np.nan
        return (w, "no_estabiliza") if return_status else w

    changes = {}
    for prev_n, n in zip(ns[:-1], ns[1:]):
        prev_v, _ = vecs[prev_n]
        v, se = vecs[n]
        if criterion == "estandarizado":
            if se is None or not np.all(np.isfinite(se)):
                changes[n] = np.inf
            else:
                changes[n] = float(np.max(np.abs(v - prev_v) / se))
        else:
            denom = np.where(np.abs(prev_v) < 1e-12, 1e-12, np.abs(prev_v))
            changes[n] = float(np.max(np.abs(v - prev_v) / denom))

    threshold = STANDARDIZED_MAX_CHANGE if criterion == "estandarizado" else STABILIZATION_REL_CHANGE
    candidates = sorted(changes.keys())
    for i, n in enumerate(candidates):
        if all(changes[m] < threshold for m in candidates[i:]):
            # Evidencia mínima: al menos 2 pasos de grilla posteriores que
            # confirman la estabilidad. Si no, es un cumplimiento "en el borde".
            status = "estabiliza" if len(candidates) - i >= 3 else "en_borde"
            return (float(n), status) if return_status else float(n)
    # Nunca estabiliza dentro de la grilla: techo de la grilla (no se extrapola mas allá de GRID_MAX).
    w = float(candidates[-1])
    return (w, "no_estabiliza") if return_status else w


def mean_annual_exceedance_frequency(returns: pd.Series, exceedance_z: float = EXCEEDANCE_Z) -> float:
    """
    f_bar_i: frecuencia anual media de excedencias |z|>2 (z respecto del
    desvío muestral completo del activo).
    """
    x = returns.dropna()
    if len(x) == 0:
        return np.nan
    z = (x - x.mean()) / x.std(ddof=0)
    n_years = len(x) / 252.0
    if n_years <= 0:
        return np.nan
    return float((z.abs() > exceedance_z).sum() / n_years)


def testing_power_window(returns: pd.Series, exceedance_z: float = EXCEEDANCE_Z) -> float:
    """
    W_T,i = 10 / f_bar_i * 252, con f_bar_i = frecuencia anual media de
    excedencias |z|>2.
    """
    f_bar = mean_annual_exceedance_frequency(returns, exceedance_z)
    if not f_bar or f_bar <= 0 or np.isnan(f_bar):
        return np.nan
    return float(10.0 / f_bar * 252.0)


def _episode_is_covered(
    returns: pd.Series,
    window: int,
    episode_dates: list[pd.Timestamp],
    detection_z: float = EPISODE_DETECTION_Z,
    tolerance_days: int = EPISODE_DATE_TOLERANCE_DAYS,
) -> bool:
    """Ver supuesto operativo en el docstring del modulo."""
    x = returns.dropna()
    if len(x) < window + 5:
        return False
    idx = x.index
    z_abs = (x - x.rolling(window).mean()) / x.rolling(window).std(ddof=0)
    exceed = (x.abs() > EXCEEDANCE_Z * x.rolling(window).std(ddof=0)).astype(float)
    intensity = exceed.rolling(window).mean()
    intensity_z = (intensity - intensity.rolling(window).mean()) / intensity.rolling(window).std(ddof=0)

    for ep_date in episode_dates:
        if ep_date < idx.min() or ep_date > idx.max():
            continue
        pos = idx.searchsorted(ep_date)
        lo = max(0, pos - tolerance_days)
        hi = min(len(idx), pos + tolerance_days + 1)
        segment = intensity_z.iloc[lo:hi]
        if (segment > detection_z).any():
            return True
    return False


def kaufman_floor_window(
    returns: pd.Series,
    episode_catalog: dict[str, list[str]] | None = None,
    grid: np.ndarray | None = None,
    event_target: int = DEFAULT_EVENT_TARGET,
) -> tuple[float, int]:
    """
    W_i^Kaufman: menor W de la grilla que cubre >= event_target de los
    episodios del catalogo.

    Devuelve (W_kaufman, episodios_cubiertos_en_ese_W).
    """
    catalog = episode_catalog or EPISODE_CATALOG
    episodes = {
        name: [pd.Timestamp(d) for d in dates] for name, dates in catalog.items()
    }
    grid = grid if grid is not None else _grid()

    best = (np.nan, 0)
    for n in sorted(grid):
        covered = sum(
            _episode_is_covered(returns, int(n), dates) for dates in episodes.values()
        )
        if covered >= event_target:
            return float(n), covered
        if covered > best[1]:
            best = (float(n), covered)
    # Ninguna ventana de la grilla alcanza el objetivo: se devuelve la mejor encontrada.
    return best


def compute_asset_window(
    ticker: str,
    returns: pd.Series,
    episode_catalog: dict[str, list[str]] | None = None,
    criterion: str = DEFAULT_WINDOW_CRITERION,
    combine: str = DEFAULT_WINDOW_COMBINE,
) -> WindowBreakdown:
    """
    W_i = max(W_E,i + W_T,i, W_i^Kaufman)   (combine="suma", especificación)
    W_i = max(W_E,i, W_T,i, W_i^Kaufman)    (combine="max" , alternativa razonable)

    `criterion` elige el criterio de estabilización de W_E (WINDOW_CRITERIA).

    W_i se elige con toda la historia disponible del activo.

    Excepción BYMA:
    si W_Kaufman con objetivo de 4 episodios supera la historia disponible,
    se relaja el objetivo a BYMA_RELAXED_EVENT_TARGET.
    """
    notes: list[str] = []
    if combine not in WINDOW_COMBINE:
        raise ValueError(f"combine desconocido: {combine}. Opciones: {WINDOW_COMBINE}")
    w_e, we_status = stabilization_window(returns, criterion=criterion, return_status=True)
    if we_status != "estabiliza":
        notes.append(f"W_E {we_status.replace('_', ' ')} (criterio '{criterion}').")
    w_t = testing_power_window(returns)

    n_obs = returns.dropna().shape[0]
    w_kaufman, covered = kaufman_floor_window(returns, episode_catalog, event_target=DEFAULT_EVENT_TARGET)
    byma_exception = False
    if (np.isnan(w_kaufman) or w_kaufman > n_obs) and ticker.upper().startswith("BYMA"):
        byma_exception = True
        w_kaufman, covered = kaufman_floor_window(
            returns, episode_catalog, event_target=BYMA_RELAXED_EVENT_TARGET
        )
        notes.append(
            f"Excepcion BYMA aplicada: objetivo relajado a {BYMA_RELAXED_EVENT_TARGET} eventos "
            f"({covered} cubiertos)."
        )
    elif np.isnan(w_kaufman):
        notes.append("W_Kaufman no alcanzo el objetivo de cobertura dentro de la grilla evaluada.")

    if combine == "suma":
        w_e_plus_t = w_e + w_t if not (np.isnan(w_e) or np.isnan(w_t)) else np.nan
        candidates = [v for v in (w_e_plus_t, w_kaufman) if not np.isnan(v)]
    else:
        candidates = [v for v in (w_e, w_t, w_kaufman) if not np.isnan(v)]
    w_final = int(round(max(candidates))) if candidates else None


    if w_final is not None and w_final > n_obs:
        margin = 100  
        capped = max(n_obs - margin, 0)
        if capped >= 250:  
            notes.append(
                f"W_i recortado de {w_final} a {capped} (tope practico = historia disponible "
                f"- {margin} ruedas) para que {ticker} no quede sin estimadores; NO es parte de "
                "la formula cerrada, es una salvaguarda de implementación."
            )
            w_final = capped
        else:
            notes.append(
                f"W_i calculado ({w_final}) supera la historia disponible ({n_obs} obs.), y el "
                "recorte practico daría una ventana por debajo del piso de la grilla (250) - se "
                f"deja W_i como esta, {ticker} quedara sin estimadores validos hasta que tenga "
                "mas historia."
            )

    return WindowBreakdown(
        ticker=ticker,
        w_stabilization=w_e,
        w_testing_power=w_t,
        w_kaufman=w_kaufman,
        w_final=w_final,
        byma_exception_applied=byma_exception,
        notes=notes,
        criterion=criterion,
        combine=combine,
        we_status=we_status,
    )
