"""
nig_model.py

Utilidades de bajo nivel para el proceso Normal Inverse Gaussian (NIG),
elegido en cierre_especificacion_estrategia.qmd (Seccion 3) como el unico
modelo de Levy definitivo del framework.

Parametrizacion (alpha, beta, delta, mu) <-> parametrizacion de scipy (a, b, loc, scale):
    alpha = a / scale      (tail heaviness)
    beta  = b / scale      (asimetria)
    delta = scale          (escala)
    mu    = loc            (locacion)

con la restriccion |beta| < alpha (scipy exige 0 < |b| < a).

Todas las funciones trabajan sobre una unica escala temporal (retornos
diarios, sin reescalar por el paso de tiempo de la medida de Levy) -
consistente con el uso que hace `cierre_especificacion_estrategia.qmd` de
`scipy.stats.norminvgauss` recalibrado directamente sobre retornos diarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, special
from scipy.stats import norminvgauss

# Bajo esta cantidad de observaciones el MLE de 4 parametros es notoriamente
# inestable; se usa como piso interno de seguridad (no es un parametro de
# la especificacion, solo una guarda numerica).
MIN_OBS_FOR_FIT = 30


@dataclass(frozen=True)
class NIGParams:
    alpha: float
    beta: float
    delta: float
    mu: float

    @property
    def sigma(self) -> float:
        """Desvio estandar del NIG: Var = delta * alpha^2 * (alpha^2-beta^2)^(-3/2)."""
        var = self.delta * self.alpha**2 * (self.alpha**2 - self.beta**2) ** (-1.5)
        return float(np.sqrt(var))


def fit_nig(window_returns: np.ndarray | "pd.Series") -> NIGParams | None:
    """
    Calibra (alpha, beta, delta, mu) por MLE via scipy.stats.norminvgauss.fit
    sobre una ventana de retornos.

    Devuelve None si la ventana es demasiado corta o si el ajuste no converge
    a parametros validos (delta>0, |beta|<alpha) - el llamador debe propagar
    esto como NaN, nunca "rellenar" con un valor arbitrario.
    """
    x = np.asarray(window_returns, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < MIN_OBS_FOR_FIT:
        return None
    try:
        a, b, loc, scale = norminvgauss.fit(x)
    except Exception:
        return None
    if scale <= 0 or not (abs(b) < a) or a <= 0:
        return None
    alpha, beta, delta, mu = a / scale, b / scale, scale, loc
    if not np.isfinite([alpha, beta, delta, mu]).all():
        return None
    return NIGParams(alpha=alpha, beta=beta, delta=delta, mu=mu)


def _levy_density(x: np.ndarray, params: NIGParams) -> np.ndarray:
    """
    Densidad de la medida de Levy del NIG: nu(x) = (delta*alpha)/(pi*|x|) * exp(beta*x) * K1(alpha*|x|).

    Se calcula via la version exponencialmente escalada del Bessel
    (scipy.special.kve(1, z) = K1(z)*exp(z)) para evitar overflow*underflow
    (exp(beta*x) -> inf mientras K1(alpha*|x|) -> 0 en las colas lejanas),
    que de otro modo produce NaN en cuadratura numerica sobre soporte infinito.
    """
    x = np.asarray(x, dtype=float)
    absx = np.abs(x)
    z = params.alpha * absx
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        val = (
            (params.delta * params.alpha)
            / (np.pi * absx)
            * np.exp(params.beta * x - z)
            * special.kve(1.0, z)
        )
    return val


def jump_intensity(params: NIGParams, threshold: float) -> float:
    """
    Intensidad esperada de saltos significativos (Seccion 3):
        lambda_hat = integral_{|x|>u} nu(x) dx,   u = threshold (>0)

    Integra ambas colas por separado (la medida no es simetrica por beta != 0)
    via cuadratura numerica (scipy.integrate.quad), ya que el NIG no tiene
    forma cerrada simple para esta integral truncada.
    """
    if threshold <= 0 or params is None:
        return np.nan
    try:
        upper, _ = integrate.quad(
            lambda x: _levy_density(x, params), threshold, np.inf, limit=200
        )
        lower, _ = integrate.quad(
            lambda x: _levy_density(x, params), -np.inf, -threshold, limit=200
        )
    except Exception:
        return np.nan
    total = upper + lower
    return float(total) if np.isfinite(total) else np.nan


def tail_quantiles(params: NIGParams, lower_q: float = 0.05, upper_q: float = 0.95):
    """Percentiles 5/95 del NIG calibrado, via CDF cerrada (norminvgauss.ppf)."""
    if params is None:
        return np.nan, np.nan
    a, b, loc, scale = (
        params.alpha * params.delta,
        params.beta * params.delta,
        params.mu,
        params.delta,
    )
    try:
        lo = norminvgauss.ppf(lower_q, a, b, loc=loc, scale=scale)
        hi = norminvgauss.ppf(upper_q, a, b, loc=loc, scale=scale)
    except Exception:
        return np.nan, np.nan
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Agregados Fase 1-2 (Entrega #4)
# --------------------------------------------------------------------------

def horizon_params(params: NIGParams, h: int) -> NIGParams:
    """
    Distribucion del retorno acumulado a h dias bajo el mismo proceso NIG.

    El NIG es cerrado bajo convolucion: la suma de h retornos diarios iid
    NIG(alpha, beta, delta, mu) es NIG(alpha, beta, h*delta, h*mu). Es la
    distribucion correcta contra la cual comparar r_{t->t+h} en el
    etiquetado de eventos (variante "horizonte" de signal_generator).
    """
    return NIGParams(alpha=params.alpha, beta=params.beta,
                     delta=params.delta * h, mu=params.mu * h)


def tail_quantiles_horizon(params: NIGParams, h: int,
                           lower_q: float = 0.05, upper_q: float = 0.95):
    """Percentiles 5/95 del retorno acumulado a h dias (NIG convolucionado)."""
    if params is None:
        return np.nan, np.nan
    return tail_quantiles(horizon_params(params, h), lower_q, upper_q)


def nig_cdf(x: float, params: NIGParams) -> float:
    a, b = params.alpha * params.delta, params.beta * params.delta
    return float(norminvgauss.cdf(x, a, b, loc=params.mu, scale=params.delta))


def prob_outside(params: NIGParams, lo: float, hi: float, h: int = 1) -> float:
    """
    P(r_{t->t+h} < lo  o  r_{t->t+h} > hi) bajo el NIG convolucionado a h dias.

    Con lo/hi = percentiles 5/95 del NIG a h dias da exactamente 0.10. Con
    lo/hi = percentiles DIARIOS (definicion de evento de la especificacion
    original) da la frecuencia de eventos que el propio modelo implica, que
    es mucho mayor que 0.10 -- ver la variante tau_method="modelo".
    """
    if params is None or not np.isfinite([lo, hi]).all():
        return np.nan
    ph = horizon_params(params, h)
    return nig_cdf(lo, ph) + 1.0 - nig_cdf(hi, ph)


def _nll(theta: np.ndarray, x: np.ndarray) -> float:
    alpha, beta, delta, mu = theta
    if delta <= 0 or alpha <= 0 or abs(beta) >= alpha:
        return np.inf
    return -float(np.sum(norminvgauss.logpdf(x, alpha * delta, beta * delta,
                                              loc=mu, scale=delta)))


def standard_errors(window_returns, params: NIGParams) -> np.ndarray:
    """
    Errores estandar asintoticos del MLE de (alpha, beta, delta, mu), via la
    inversa del Hessiano numerico (diferencias centrales) de la
    log-verosimilitud negativa en el optimo.

    Se usa en el criterio de estabilizacion "estandarizado" de
    window_selection.py: medir el cambio de cada parametro en unidades de su
    propio error estandar evita el problema de dividir por un parametro
    cercano a cero (mu, y a veces beta), que hace que el criterio relativo de
    la especificacion casi nunca se cumpla.

    Devuelve un array de 4 NaN si el Hessiano no es definido positivo.
    """
    x = np.asarray(window_returns, dtype=float)
    x = x[~np.isnan(x)]
    theta = np.array([params.alpha, params.beta, params.delta, params.mu])
    steps = 1e-4 * np.maximum(np.abs(theta), np.array([1e-2, 1e-2, 1e-5, 1e-5]))
    k = len(theta)
    hess = np.zeros((k, k))
    f0 = _nll(theta, x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = steps[i]
            ej = np.zeros(k); ej[j] = steps[j]
            if i == j:
                val = (_nll(theta + ei, x) - 2 * f0 + _nll(theta - ei, x)) / steps[i] ** 2
            else:
                val = (
                    _nll(theta + ei + ej, x) - _nll(theta + ei - ej, x)
                    - _nll(theta - ei + ej, x) + _nll(theta - ei - ej, x)
                ) / (4 * steps[i] * steps[j])
            hess[i, j] = hess[j, i] = val
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)
    diag = np.diag(cov)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
        return np.full(k, np.nan)
    return np.sqrt(diag)
