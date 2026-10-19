"""Diagnóstico estadístico previo a la calibración (prometido en Entrega #2, sec. 11).

Sobre log_return diario de cada backtest_<TICKER>_BA.csv (período evaluado, es decir,
sin el calentamiento de la ventana rodante):
  - ADF (H0: raíz unitaria) y KPSS (H0: estacionariedad), sobre retornos y sobre el
    log-precio reconstruido (cumsum) como contraste.
  - Jarque-Bera y Shapiro-Wilk (H0: normalidad), asimetría y curtosis en exceso.
  - Ljung-Box sobre retornos y sobre retornos al cuadrado (dependencia / agrupamiento).
  - QQ plot contra la normal.
"""
import glob, os, re, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SRC = os.environ.get("SRC", ".")
OUT = "diag_out"; os.makedirs(OUT, exist_ok=True)
rows = []; series = {}
for f in sorted(glob.glob(os.path.join(SRC, "*backtest_*_BA.csv"))):
    t = re.search(r"backtest_(\w+)_BA", f).group(1)
    d = pd.read_csv(f, parse_dates=["Date"], index_col="Date")
    r = d["log_return"].dropna(); series[t] = r
    p = r.cumsum()
    adf_r = adfuller(r, regression="c", autolag="AIC")
    adf_p = adfuller(p, regression="c", autolag="AIC")
    kp_r = kpss(r, regression="c", nlags="auto")
    kp_p = kpss(p, regression="c", nlags="auto")
    jb = stats.jarque_bera(r)
    sw = stats.shapiro(r)
    lb1 = acorr_ljungbox(r, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb2 = acorr_ljungbox(r**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    rows.append(dict(activo=t, n=len(r), desde=r.index[0].date(), hasta=r.index[-1].date(),
        media_diaria=r.mean(), desvio_diario=r.std(), asimetria=stats.skew(r),
        curtosis_exceso=stats.kurtosis(r),
        adf_ret_stat=adf_r[0], adf_ret_p=adf_r[1], adf_logp_stat=adf_p[0], adf_logp_p=adf_p[1],
        kpss_ret_stat=kp_r[0], kpss_ret_p=kp_r[1], kpss_logp_stat=kp_p[0], kpss_logp_p=kp_p[1],
        jb_stat=jb.statistic, jb_p=jb.pvalue, sw_stat=sw.statistic, sw_p=sw.pvalue,
        lb10_ret_p=lb1, lb10_ret2_p=lb2))
res = pd.DataFrame(rows).set_index("activo")
res.to_csv(f"{OUT}/diagnostico_series.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(res.T.to_string(float_format=lambda x: f"{x:.4g}"))

fig, axs = plt.subplots(1, len(series), figsize=(3.2*len(series), 3.4))
for ax, (t, r) in zip(np.atleast_1d(axs), series.items()):
    sm.qqplot(r, line="s", ax=ax, markersize=2); ax.set_title(t); ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("QQ plot de log-retornos diarios contra la normal"); fig.tight_layout()
fig.savefig(f"{OUT}/qqplots.png", dpi=150)
