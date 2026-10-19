"""
protection.py  (Entrega #4, Fase 4)

"¿La estrategia protege cuando se decide con la información disponible?"

Toma las carteras de comparators.py (todas con el mismo motor, costos y
fechas) y responde con cinco bloques de evidencia:

1. Segmentación (estabilidad_subperiodos_regimenes.qmd)
   - dentro / fuera de episodios del catálogo (±10 ruedas hábiles de cada
     fecha);
   - régimen de volatilidad: vol de 60 ruedas del benchmark EW contra su
     mediana. Es una clasificación EX POST, para atribuir resultados, no para
     generar señal;
   - tabla por episodio (retorno acumulado y drawdown dentro de la ventana);
   - leave-one-episode-out: ¿la diferencia dentro de episodios depende de un
     solo episodio?

2. Riesgo de cola (riesgo_cola_var_es.qmd)
   VaR y ES históricos rolling OUT-OF-SAMPLE (ventana de 252 ruedas hasta
   t-1), al 95% y 99%, con los tests de Kupiec (cobertura), Christoffersen
   (independencia; razón pi11/pi01 = magnitud del clustering) y cobertura
   condicional.

3. Bootstrap estacionario por bloques (Politis-Romano), pareado por fecha:
   intervalos de confianza de la diferencia de Sharpe, retorno, drawdown y
   Calmar entre la estrategia y cada comparador.

4. Deflated Sharpe Ratio (Bailey y López de Prado, 2014): ¿el Sharpe sigue
   siendo significativo después de descontar cuántas configuraciones se
   probaron?

5. Probability of Backtest Overfitting (Bailey, Borwein, López de Prado y
   Zhu, 2017), por validación cruzada combinatoria simétrica (CSCV).

Los bloques 4 y 5 necesitan los retornos de todas las configuraciones
probadas: `run_configuration_grid` las corre reutilizando las ventanas ya
calculadas (window_criteria_comparison.csv) y la caché del NIG.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from performance_metrics import max_drawdown, summary
from window_selection import EPISODE_CATALOG

TRADING_DAYS = 252
EPISODE_WINDOW_DAYS = 10     # ±10 ruedas hábiles (estabilidad_subperiodos_regimenes.qmd)
REGIME_VOL_WINDOW = 60       # vol de 60 ruedas para el régimen (mismo .qmd)
VAR_WINDOW = 252             # riesgo_cola_var_es.qmd
VAR_LEVELS = (0.95, 0.99)
BOOTSTRAP_BLOCK = 20         # largo medio de bloque (ruedas)
BOOTSTRAP_REPS = 2000
EULER_GAMMA = 0.5772156649


# --------------------------------------------------------------------------
# 1. Segmentación
# --------------------------------------------------------------------------

def episode_mask(index: pd.DatetimeIndex, catalog: dict | None = None,
                 window: int = EPISODE_WINDOW_DAYS, only: list[str] | None = None) -> pd.Series:
    """True en las ruedas a ±`window` ruedas hábiles de alguna fecha del catálogo."""
    catalog = catalog or EPISODE_CATALOG
    mask = np.zeros(len(index), dtype=bool)
    for name, dates in catalog.items():
        if only is not None and name not in only:
            continue
        for d in dates:
            pos = index.searchsorted(pd.Timestamp(d))
            if pos >= len(index) or abs((index[min(pos, len(index) - 1)] - pd.Timestamp(d)).days) > 7:
                continue   # fecha fuera del período evaluado
            mask[max(0, pos - window): min(len(index), pos + window + 1)] = True
    return pd.Series(mask, index=index)


def volatility_regime(reference_returns: pd.Series, window: int = REGIME_VOL_WINDOW) -> pd.Series:
    """'alta' / 'baja' según la vol de `window` ruedas contra su mediana (ex post)."""
    vol = reference_returns.rolling(window).std()
    return pd.Series(np.where(vol > vol.median(), "alta", "baja"), index=reference_returns.index).where(vol.notna())


def _segment_row(r: pd.Series, risk_free_rate=None) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"días": len(r)}
    s = summary(r, risk_free_rate=risk_free_rate)
    return {"días": len(r), "retorno_anual": s["Retorno anualizado"], "volatilidad": s["Volatilidad anualizada"],
            "sharpe": s["Sharpe"], "max_drawdown": s["Max Drawdown"],
            "retorno_medio_diario": r.mean(), "peor_dia": r.min()}


def segmented_metrics(portfolios: dict[str, pd.DataFrame], risk_free_rate=None) -> pd.DataFrame:
    """Métricas de cada cartera dentro/fuera de episodios y por régimen de volatilidad."""
    ref = portfolios["EW 100%"]["net_return"]
    idx = ref.index
    segments = {
        "dentro de episodio": episode_mask(idx),
        "fuera de episodio": ~episode_mask(idx),
    }
    regime = volatility_regime(ref)
    segments["vol alta"] = regime.eq("alta")
    segments["vol baja"] = regime.eq("baja")
    rows = []
    for seg, m in segments.items():
        for name, df in portfolios.items():
            r = df["net_return"].reindex(idx)[m.reindex(idx).fillna(False).to_numpy()]
            rows.append({"segmento": seg, "cartera": name, **_segment_row(r, risk_free_rate)})
    return pd.DataFrame(rows)


def per_episode_table(portfolios: dict[str, pd.DataFrame], catalog: dict | None = None,
                      window: int = EPISODE_WINDOW_DAYS) -> pd.DataFrame:
    """
    Por episodio del catálogo que cae en el período evaluado: retorno
    acumulado y drawdown máximo de cada cartera dentro de la ventana
    [primera fecha - window, última fecha + window].
    """
    catalog = catalog or EPISODE_CATALOG
    idx = portfolios["Estrategia"].index
    rows = []
    for name, dates in catalog.items():
        ds = [pd.Timestamp(d) for d in dates]
        if ds[-1] < idx[0] or ds[0] > idx[-1]:
            continue
        a = max(0, idx.searchsorted(ds[0]) - window)
        b = min(len(idx) - 1, idx.searchsorted(ds[-1]) + window)
        start, end = idx[a], idx[b]
        for pname, df in portfolios.items():
            r = df["net_return"].loc[start:end].dropna()
            if r.empty:
                continue
            nav = (1 + r).cumprod()
            rows.append({"episodio": name, "desde": start.date(), "hasta": end.date(), "días": len(r),
                         "cartera": pname, "retorno_acumulado": nav.iloc[-1] - 1,
                         "max_drawdown": float((nav / nav.cummax().clip(lower=1.0) - 1).min())})
    return pd.DataFrame(rows)


def leave_one_episode_out(portfolios: dict[str, pd.DataFrame], versus: str = "EW 100%",
                          catalog: dict | None = None, window: int = EPISODE_WINDOW_DAYS,
                          risk_free_rate=None) -> pd.DataFrame:
    """
    Métricas DENTRO de episodios con todos los episodios y excluyendo uno por
    vez. Si la ventaja de la estrategia desaparece al sacar un episodio, la
    evidencia depende de ese episodio.
    """
    catalog = catalog or EPISODE_CATALOG
    idx = portfolios["Estrategia"].index
    present = [n for n in catalog if episode_mask(idx, catalog, window, only=[n]).any()]
    variants = {"todos": present} | {f"sin {n}": [m for m in present if m != n] for n in present}
    rows = []
    for label, keep in variants.items():
        m = episode_mask(idx, catalog, window, only=keep).to_numpy()
        s = portfolios["Estrategia"]["net_return"][m]
        b = portfolios[versus]["net_return"].reindex(idx)[m]
        if len(s) < 5:
            continue
        rows.append({"episodios": label, "días": int(m.sum()),
                     "sharpe_estrategia": summary(s, risk_free_rate=risk_free_rate)["Sharpe"],
                     f"sharpe_{versus}": summary(b, risk_free_rate=risk_free_rate)["Sharpe"],
                     "retorno_medio_estrategia": s.mean(), f"retorno_medio_{versus}": b.mean(),
                     "dif_retorno_medio_diario": s.mean() - b.mean(),
                     "peor_dia_estrategia": s.min(), f"peor_dia_{versus}": b.min()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. VaR / ES y validación
# --------------------------------------------------------------------------

def var_es_rolling(returns: pd.Series, window: int = VAR_WINDOW, alpha: float = 0.95) -> tuple[pd.Series, pd.Series]:
    """VaR y ES históricos (pérdida, positivos) estimados con [t-window, t-1]."""
    r = returns.dropna()
    q = r.rolling(window).quantile(1 - alpha).shift(1)
    var = -q

    def _es(x):
        cut = np.quantile(x, 1 - alpha)
        return -x[x <= cut].mean()

    es = r.rolling(window).apply(_es, raw=True).shift(1)
    return var, es


def kupiec_test(exceptions: pd.Series, alpha: float) -> dict:
    x = exceptions.dropna().astype(int)
    T, n = len(x), int(x.sum())
    p = 1 - alpha
    if T == 0:
        return {"n": 0}
    phat = n / T
    ll0 = xlogy(T - n, 1 - p) + xlogy(n, p)
    ll1 = xlogy(T - n, 1 - phat) + xlogy(n, phat)
    lr = -2 * (ll0 - ll1)
    return {"n": T, "excepciones": n, "tasa": phat, "tasa_nominal": p,
            "LR": lr, "p_value": float(stats.chi2.sf(lr, 1))}


def christoffersen_test(exceptions: pd.Series) -> dict:
    x = exceptions.dropna().astype(int).to_numpy()
    if len(x) < 2:
        return {}
    a, b = x[:-1], x[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (ll0 - ll1)
    return {"pi01": pi01, "pi11": pi11, "razon_pi11_pi01": pi11 / pi01 if pi01 > 0 else np.nan,
            "LR": lr, "p_value": float(stats.chi2.sf(lr, 1))}


def tail_risk_table(portfolios: dict[str, pd.DataFrame], window: int = VAR_WINDOW,
                    levels=VAR_LEVELS) -> pd.DataFrame:
    """VaR/ES rolling out-of-sample + Kupiec + Christoffersen + cobertura condicional, por cartera."""
    rows = []
    for name, df in portfolios.items():
        r = df["net_return"].dropna()
        for a in levels:
            var, es = var_es_rolling(r, window, a)
            valid = var.notna()
            exc = (r[valid] < -var[valid])
            kup = kupiec_test(exc, a)
            chr_ = christoffersen_test(exc)
            lr_cc = kup.get("LR", np.nan) + chr_.get("LR", np.nan)
            loss_on_exc = -r[valid][exc]
            rows.append({
                "cartera": name, "nivel": a, "días_evaluados": int(valid.sum()),
                "VaR_medio": var[valid].mean(), "ES_medio": es[valid].mean(),
                "VaR_empirico": -np.quantile(r, 1 - a), "ES_empirico": -r[r <= np.quantile(r, 1 - a)].mean(),
                "excepciones": kup.get("excepciones"), "tasa_excepciones": kup.get("tasa"),
                "kupiec_p": kup.get("p_value"), "christoffersen_p": chr_.get("p_value"),
                "razon_pi11_pi01": chr_.get("razon_pi11_pi01"),
                "cobertura_condicional_p": float(stats.chi2.sf(lr_cc, 2)) if np.isfinite(lr_cc) else np.nan,
                "perdida_media_en_excepcion": loss_on_exc.mean() if len(loss_on_exc) else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Bootstrap estacionario por bloques
# --------------------------------------------------------------------------

def stationary_bootstrap_indices(T: int, reps: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Índices (reps x T) del bootstrap estacionario de Politis y Romano (1994)."""
    idx = np.empty((reps, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, reps)
    new_block = rng.random((reps, T)) < 1.0 / block
    jumps = rng.integers(0, T, (reps, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def _metrics_matrix(R: np.ndarray, RF: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Métricas por fila de una matriz de retornos simples (reps x T).
    `RF` (misma forma que R) es la tasa libre de riesgo diaria remuestreada con
    los mismos índices; solo afecta al Sharpe."""
    T = R.shape[1]
    X = R if RF is None else R - RF
    mean, sd = X.mean(axis=1), X.std(axis=1, ddof=1)
    logs = np.log1p(R)
    ann = np.exp(logs.sum(axis=1) * TRADING_DAYS / T) - 1
    nav = np.exp(np.cumsum(logs, axis=1))
    peak = np.maximum(np.maximum.accumulate(nav, axis=1), 1.0)
    mdd = (nav / peak - 1).min(axis=1)
    return {"sharpe": mean / sd * np.sqrt(TRADING_DAYS), "retorno_anual": ann, "max_drawdown": mdd,
            "calmar": ann / np.abs(mdd)}


def bootstrap_differences(strategy: pd.Series, comparator: pd.Series, reps: int = BOOTSTRAP_REPS,
                          block: float = BOOTSTRAP_BLOCK, seed: int = 0,
                          risk_free: pd.Series | None = None) -> pd.DataFrame:
    """
    Diferencia estrategia − comparador (Sharpe, retorno anual, max drawdown,
    Calmar), con bootstrap estacionario PAREADO por fecha: se remuestrean los
    mismos días para ambas series, así se conserva su correlación.

    En max_drawdown, una diferencia positiva significa que la estrategia cae
    menos. `p_dif<=0` es la proporción de réplicas en que la estrategia no
    supera al comparador (p-valor unilateral de "la estrategia es mejor").
    """
    df = pd.concat({"s": strategy, "c": comparator}, axis=1).dropna()
    S, C = df["s"].to_numpy(), df["c"].to_numpy()
    RF = None if risk_free is None else risk_free.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(df), reps, block, rng)
    rfi = None if RF is None else RF[idx]
    ms, mc = _metrics_matrix(S[idx], rfi), _metrics_matrix(C[idx], rfi)
    rf1 = None if RF is None else RF[None, :]
    obs_s, obs_c = _metrics_matrix(S[None, :], rf1), _metrics_matrix(C[None, :], rf1)
    rows = []
    for k in ms:
        d = ms[k] - mc[k]
        rows.append({"métrica": k, "estrategia": obs_s[k][0], "comparador": obs_c[k][0],
                     "diferencia": obs_s[k][0] - obs_c[k][0],
                     "IC95_inf": np.quantile(d, 0.025), "IC95_sup": np.quantile(d, 0.975),
                     "p_dif<=0": float((d <= 0).mean())})
    return pd.DataFrame(rows)


def bootstrap_table(portfolios: dict[str, pd.DataFrame],
                    versus=("EW 100%", "Exposición fija (media ex post)", "Vol targeting"),
                    reps: int = BOOTSTRAP_REPS, block: float = BOOTSTRAP_BLOCK, seed: int = 0,
                    risk_free: pd.Series | None = None) -> pd.DataFrame:
    out = []
    for v in versus:
        if v in portfolios:
            t = bootstrap_differences(portfolios["Estrategia"]["net_return"], portfolios[v]["net_return"],
                                      reps, block, seed, risk_free)
            t.insert(0, "comparador_nombre", v)
            out.append(t)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# 4-5. Múltiples configuraciones: Deflated Sharpe y PBO
# --------------------------------------------------------------------------

def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: np.ndarray, n_trials: int | None = None) -> dict:
    """
    Deflated Sharpe Ratio (Bailey y López de Prado, 2014).

    `trial_sharpes`: Sharpe POR PERÍODO (diario, sin anualizar) de todas las
    configuraciones probadas. SR0 es el Sharpe máximo esperado por azar entre
    N pruebas independientes; DSR = P(SR verdadero > SR0), corrigiendo por
    asimetría y curtosis de los retornos elegidos.
    """
    r = returns.dropna().to_numpy()
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    trial = np.asarray(trial_sharpes, dtype=float)
    trial = trial[np.isfinite(trial)]
    N = int(n_trials or len(trial))
    var_sr = trial.var(ddof=1) if len(trial) > 1 else 0.0
    if N > 1:
        sr0 = np.sqrt(var_sr) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / N)
                                 + EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * np.e)))
    else:
        sr0 = 0.0
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    psr0 = stats.norm.cdf(sr * np.sqrt(T - 1) / denom)   # Probabilistic Sharpe vs. 0
    return {"T": T, "N_pruebas": N, "sharpe_diario": sr, "sharpe_anual": sr * np.sqrt(TRADING_DAYS),
            "SR0_diario": sr0, "SR0_anual": sr0 * np.sqrt(TRADING_DAYS),
            "asimetria": skew, "curtosis": kurt, "PSR_vs_0": psr0, "DSR": float(stats.norm.cdf(z))}


def probability_backtest_overfitting(returns_matrix: pd.DataFrame, n_blocks: int = 16) -> dict:
    """
    PBO por CSCV (Bailey, Borwein, López de Prado y Zhu, 2017).

    `returns_matrix`: fechas x configuraciones, sin NaN (fechas comunes). Se
    parte en `n_blocks` bloques contiguos; para cada combinación de la mitad de
    los bloques como "in-sample" se elige la configuración de mayor Sharpe IS y
    se mira su ranking relativo OOS (w). PBO = P(logit(w) <= 0), es decir, la
    probabilidad de que la mejor configuración IS quede por debajo de la
    mediana OOS.
    """
    M = returns_matrix.dropna().to_numpy()
    T, N = M.shape
    if N < 2 or T < n_blocks * 5:
        return {"PBO": np.nan, "N_config": N, "T": T}
    bounds = np.linspace(0, T, n_blocks + 1).astype(int)
    s1 = np.array([M[bounds[k]:bounds[k + 1]].sum(axis=0) for k in range(n_blocks)])
    s2 = np.array([(M[bounds[k]:bounds[k + 1]] ** 2).sum(axis=0) for k in range(n_blocks)])
    cnt = np.diff(bounds)

    def sharpe(sel):
        n = cnt[sel].sum()
        m = s1[sel].sum(axis=0) / n
        v = s2[sel].sum(axis=0) / n - m ** 2
        return m / np.sqrt(np.maximum(v, 1e-18))

    lambdas, degr = [], []
    all_blocks = np.arange(n_blocks)
    for is_blocks in combinations(all_blocks, n_blocks // 2):
        is_sel = np.array(is_blocks)
        oos_sel = np.setdiff1d(all_blocks, is_sel)
        sr_is, sr_oos = sharpe(is_sel), sharpe(oos_sel)
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 1          # 1..N
        w = rank / (N + 1)
        lambdas.append(np.log(w / (1 - w)))
        degr.append((sr_is[best], sr_oos[best]))
    lambdas = np.array(lambdas)
    degr = np.array(degr)
    slope = np.polyfit(degr[:, 0], degr[:, 1], 1)[0] if len(degr) > 2 else np.nan
    return {"PBO": float((lambdas <= 0).mean()), "N_config": N, "T": T, "combinaciones": len(lambdas),
            "logit_mediano": float(np.median(lambdas)),
            "pendiente_IS_OOS": float(slope)}


def windows_from_criteria_table(table: pd.DataFrame) -> dict[tuple[str, str], dict[str, int]]:
    """{(criterio, combinación): {ticker: W_i}} desde window_criteria_comparison.csv."""
    out = {}
    for (c, m), g in table.groupby(["criterio_W", "combinacion"]):
        out[(c, m)] = {r["activo"]: int(r["W_i"]) for _, r in g.iterrows()}
    return out


def run_configuration_grid(
    returns_by_asset: dict[str, pd.Series],
    windows_by_criterion: dict[tuple[str, str], dict[str, int]],
    event_tau=(("diaria", "excedencias"), ("diaria", "modelo"),
               ("horizonte", "excedencias"), ("horizonte", "modelo")),
    refit_step: int = 5,
    verbose: bool = True,
    price_data: dict | None = None,
    cash_rate: pd.Series | None = None,
    **config_kwargs,
) -> pd.DataFrame:
    """
    Corre la estrategia para todas las combinaciones (criterio de W) x
    (evento, tau) y devuelve los retornos netos diarios de la cartera de cada
    una (columnas = configuraciones). Reutiliza W_i ya calculados y la caché
    del NIG; el logit se reajusta cada `refit_step` ruedas para que la grilla
    sea viable (solo afecta a esta evaluación de sobreajuste).
    """
    from main import PipelineConfig, run_pipeline   # import diferido: evita ciclo

    cols = {}
    for (crit, comb), windows in windows_by_criterion.items():
        for ev, tau in event_tau:
            label = f"{crit}/{comb}|{ev}|{tau}"
            cfg = PipelineConfig(window_criterion=crit, window_combine=comb, event_definition=ev,
                                 tau_method=tau, windows=windows, refit_step=refit_step, **config_kwargs)
            res = run_pipeline(returns_by_asset, cfg, verbose=False, price_data=price_data, cash_rate=cash_rate)
            if res["portfolios"]:
                cols[label] = res["portfolios"]["Estrategia"]["net_return"]
            if verbose:
                n = len(cols.get(label, []))
                print(f"  {label}: {n} días")
    return pd.DataFrame(cols)


def overfitting_summary(grid_returns: pd.DataFrame, chosen: str, min_days: int = 2 * TRADING_DAYS,
                        n_blocks: int = 16, risk_free: pd.Series | None = None) -> dict:
    """
    DSR de la configuración elegida y PBO, ambos sobre las FECHAS COMUNES de
    las configuraciones con al menos `min_days` de historia.

    Por qué el período común: el DSR usa la dispersión de los Sharpe de las
    pruebas. Si cada configuración cubre un período distinto (en la grilla
    real, de 48 días a 7 años), esa dispersión mezcla el efecto de la
    configuración con el del período, y el umbral SR0 se infla. N sigue
    siendo el total de configuraciones probadas (criterio conservador).
    `dsr_periodos_distintos` conserva el cálculo ingenuo como referencia.
    """
    if risk_free is not None:
        # DSR y PBO sobre retornos en EXCESO de la tasa libre de riesgo.
        grid_returns = grid_returns.sub(risk_free.reindex(grid_returns.index).fillna(0.0), axis=0)
    daily_sr = grid_returns.apply(lambda c: c.dropna().mean() / c.dropna().std(ddof=1)
                                  if c.dropna().shape[0] > 20 else np.nan)
    n_total = grid_returns.shape[1]
    dsr_naive = deflated_sharpe_ratio(grid_returns[chosen], daily_sr.to_numpy(), n_trials=n_total)
    long_enough = [c for c in grid_returns if grid_returns[c].notna().sum() >= min_days]
    common = grid_returns[long_enough].dropna()
    if chosen in common and len(common) > 20 and len(long_enough) > 1:
        sr_common = (common.mean() / common.std(ddof=1)).to_numpy()
        dsr = deflated_sharpe_ratio(common[chosen], sr_common, n_trials=n_total)
    else:
        dsr = dsr_naive
    pbo = probability_backtest_overfitting(common, n_blocks) if len(long_enough) > 1 else {"PBO": np.nan}
    pbo["configs_en_PBO"] = len(long_enough)
    pbo["fechas_comunes"] = f"{common.index[0].date()} → {common.index[-1].date()}" if len(common) else ""
    return {"dsr": dsr, "dsr_periodos_distintos": dsr_naive, "pbo": pbo,
            "sharpe_anual_por_config": (daily_sr * np.sqrt(TRADING_DAYS)).sort_values(ascending=False)}
