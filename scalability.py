"""
Escalabilidad

Depende de transaction_costs.py y robustness.py (misma carpeta). No modifica nada del pipeline.
AUM en USD (se convierte a pesos con el USD/ARS oficial de cada fecha, `fx`).

    import scalability as sc
    out = sc.run_all(res, price_data, fx_oficial)          # ~1-2 min
    out["escalera"]       # desempeño neto de impacto y liquidez por AUM y b
    out["ejecucion"]      # liquidez por AUM: % sobre el tope, días para completar, con y sin BYMA
    out["punto_quiebre"]  # AUM al que se pierde cada umbral (interpolado)
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import liquidity as lq
import transaction_costs as tc

AUMS_USD = (1e5, 2.5e5, 1e6, 2.5e6, 5e6, 1e7, 2e7, 5e7, 1e8)


def _exec_stats(res, price_data, fx, aum, b, cap, desde):
    d = tc.impact_diagnostic(res, price_data, fx, aum=aum, b=b, cap=cap)["ordenes"].dropna(subset=["participacion"])
    d["dias"] = np.ceil(d["participacion"] / cap).clip(lower=1)   # ruedas necesarias para completar al tope
    w = d["fraccion_nav"]
    rec = d[pd.to_datetime(d["fecha"]) >= pd.Timestamp(desde)] if desde else d
    sb = d[d["activo"] != "BYMA.BA"]
    return {
        "ordenes": len(d),
        "pct_sobre_tope": float(d["tope"].mean()),
        "pct_sobre_tope_sin_BYMA": float(sb["tope"].mean()) if len(sb) else np.nan,
        "pct_volumen_sobre_tope": float(w[d["tope"]].sum() / w.sum()),   # parte del monto operado en órdenes que no entran en una rueda
        "dias_p95": float(d["dias"].quantile(0.95)),
        "dias_max": float(d["dias"].max()),
        "pct_sobre_tope_desde": float(rec["tope"].mean()) if len(rec) else np.nan,
    }


def aum_ladder(res, price_data, fx, aums=AUMS_USD, b_values=(tc.IMPACT_B, 3 * tc.IMPACT_B),
               cap=lq.ADV_CAP_PCT, desde="2023-01-01"):
    """Una fila por AUM y b: desempeño neto de impacto (estrategia, EW, fija) + liquidez de las órdenes."""
    imp = tc.market_impact_sensitivity(res, price_data, aums, b_values, cap=cap, fx=fx)
    rows = []
    for (aum, b), r in imp.iterrows():
        extra = _exec_stats(res, price_data, fx, aum, b, cap, desde) if aum else {}
        rows.append(extra)
    ex = pd.DataFrame(rows, index=imp.index)
    keep = ["sharpe", "sharpe_EW", "dif_sharpe_vs_EW", "dif_sharpe_vs_fija", "retorno_anual",
            "max_drawdown", "dif_mdd_vs_EW", "costo_total_pp", "costo_total_pp_EW",
            "impacto_medio_pb", "d_sharpe_vs_base", "d_retorno_vs_base"]
    out = imp[keep].join(ex.drop(columns=["ordenes"], errors="ignore"))
    out.attrs["desde"] = desde
    return out


def breakeven(ladder: pd.DataFrame, umbrales=(0.01, 0.025, 0.05)):
    """AUM (USD, interpolado en logaritmo) al que el Sharpe cae `u` por debajo de la base, para cada b."""
    rows = []
    for b, g in ladder.drop(index=0.0, level="aum", errors="ignore").groupby(level="b"):
        a = g.index.get_level_values("aum").to_numpy(float)
        loss = -g["d_sharpe_vs_base"].to_numpy(float)
        for u in umbrales:
            if loss.max() < u:   r = f"> {a.max():,.0f}"
            elif loss.min() >= u: r = f"< {a.min():,.0f}"
            else:                r = f"{np.exp(np.interp(u, loss, np.log(a))):,.0f}"
            rows.append({"b": b, "caida_sharpe": u, "aum_usd": r})
    return pd.DataFrame(rows)


def run_all(res, price_data, fx, aums=AUMS_USD, prefix="escala_", save=True, verbose=True, desde="2023-01-01"):
    log = print if verbose else (lambda *a, **k: None)
    out = {}
    t0 = time.time()
    lad = aum_ladder(res, price_data, fx, aums, desde=desde)
    out["escalera"] = lad
    out["ejecucion"] = lad.loc[lad.index.get_level_values("b") == tc.IMPACT_B,
                               ["pct_sobre_tope", "pct_sobre_tope_sin_BYMA", "pct_volumen_sobre_tope",
                                "dias_p95", "dias_max", "pct_sobre_tope_desde"]]
    out["punto_quiebre"] = breakeven(lad)
    log(f"Escalabilidad: {time.time() - t0:.0f} s")
    if save:
        for k, df in out.items():
            df.to_csv(f"{prefix}{k}.csv")
    return out
