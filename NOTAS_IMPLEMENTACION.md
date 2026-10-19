# Notas de implementación — motor de backtesting

Este documento acompaña a `data_loader.py`, `window_selection.py`, `nig_model.py`, `levy_estimators.py`, `signal_generator.py`, `backtest_engine.py`, `performance_metrics.py` y `main.py`.

> **Nota sobre las cifras.** Las secciones anteriores son el registro histórico de la construcción (corridas con paso 5 y caja al 0%); sus números **no** son los finales. Las cifras definitivas son las de la sección «Corrida final» al final de este documento (paso 1, caja y Sharpe con la tasa del pase a 1 día).

---

## Registro de cambios — Entrega #4

### Fase 0: base limpia (errores que afectaban cualquier corrida)

| # | Archivo | Problema en la Entrega #3 | Corrección |
|---|---|---|---|
| 0.1 | `window_selection.py` | El catálogo tenía `2025-08-09` (sábado; día y mes invertidos de `2025-09-08`) y una fecha extra, `2024-07-24`: 19 fechas en lugar de las 18 documentadas. Afecta directamente a W_i^Kaufman. | Catálogo de 18 fechas, todas ruedas hábiles. 2024-07-24 queda fuera; si se quiere conservar, hay que agregarla al catálogo de episodios. |
| 0.2 | `backtest_engine.py` | Se cobraba el costo **round-trip** (226–258 pb) por cada cambio de exposición **one-way**: el doble del costo correcto. | Costo **por lado** = comisión + derecho de mercado (66.55 pb) + **medio** spread de Corwin-Schultz. Un ciclo 0 → 100% → 0 sigue costando exactamente un round-trip. |
| 0.3 | `backtest_engine.py` | El NAV componía **log-retornos** como si fueran simples (`(1 + r_log).cumprod()`), lo que sesga hacia abajo tanto a la estrategia como al benchmark. | El motor recibe log-retornos y convierte internamente (`r_simple = exp(r_log) − 1`). `net_return`, `nav` y el benchmark salen en retornos simples, que es lo que esperan las métricas. |
| 0.4 | `data_loader.py` | Usaba `Close` con el default de yfinance (`auto_adjust` cambió de default entre versiones); la especificación usa `Adj Close` con `auto_adjust=False`. | `auto_adjust=False` explícito; `log_return` sobre `Adj Close`. `Close`, `High`, `Low` y `Volume` quedan sin ajustar para ADV y Corwin-Schultz (Fase 5). Falla con un error explícito si falta `Adj Close`. |
| 0.5 | `performance_metrics.py` | (a) El Max Drawdown no contaba una caída inmediata desde el capital inicial. (b) El Sortino usaba el desvío de los retornos negativos solamente, que no es la definición estándar. | (a) Drawdown medido también contra NAV = 1. (b) Desvío a la baja estándar: √mean(min(r − rf, 0)²) sobre todas las observaciones. Se agrega la fila `Días` al resumen. |
| 0.6 | Docstrings y estas notas | Comentarios que decían que el costo seguía en 50 pb; referencia a `Z_SCORE_MIN_PERIODS_FRACTION`, que ya no existe; `signal_generator.py` afirmaba ausencia de look-ahead. | Actualizados. `signal_generator.py` advierte explícitamente que el ajuste del logit todavía usa toda la muestra (se corrige en la Fase 1). |

#### Costos por activo (pb)

Costo **por unidad de turnover one-way** = fijo por lado + medio spread. El round-trip es el doble.

| Activo | Fijo por lado | ½ spread medio | **Por lado (default)** | Round-trip | ½ spread p95 | Por lado p95 | Round-trip p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GGAL | 66.55 | 46.75 | **113.30** | 226.6 | 178.70 | 245.25 | 490.5 |
| PAMP | 66.55 | 55.30 | **121.85** | 243.7 | 183.80 | 250.35 | 500.7 |
| YPFD | 66.55 | 45.55 | **112.10** | 224.2 | 172.15 | 238.70 | 477.4 |
| BMA | 66.55 | 61.00 | **127.55** | 255.1 | 212.00 | 278.55 | 557.1 |
| BYMA | 66.55 | 62.50 | **129.05** | 258.1 | 200.20 | 266.75 | 533.5 |

Fuentes: comisión 0.50% + IVA (IOL, supuesto conservador declarado), derecho de mercado 5 pb + IVA (BYMA-COM18777, desde 1-oct-2025), spread de Corwin-Schultz (media y p95). Se generan con `backtest_engine.cost_table()` y `main.py` los guarda en `cost_table_bps.csv`.

#### Efecto de la Fase 0 sobre las cifras de la Entrega #3 (solo ilustrativo)

Reprocesando los CSV de la Entrega #3 (mismos log-retornos y **mismas exposiciones**, que siguen teniendo look-ahead) con el motor corregido:

| Métrica | E3 Estrategia | E3 Benchmark | F0 Estrategia | F0 Benchmark |
|---|---:|---:|---:|---:|
| Retorno anualizado | 26.16% | 47.92% | 43.43% | 58.66% |
| Volatilidad anualizada | 29.25% | 31.01% | 29.35% | 31.38% |
| Sharpe | 0.938 | 1.415 | 1.374 | 1.626 |
| Sortino | 1.897 | 2.931 | 2.398 | 2.904 |
| Max Drawdown | −15.76% | −15.85% | −14.60% | −15.07% |
| Calmar | 1.660 | 3.024 | 2.975 | 3.892 |

Descomposición para la estrategia: la corrección de la contabilidad (retornos simples) lleva el retorno anualizado de 26.2% a 34.7%, y el costo por lado lo lleva de 34.7% a 43.4%. El benchmark solo se ve afectado por la contabilidad. **Estas cifras no deben reportarse**: las exposiciones todavía tienen look-ahead (Fase 1) y el período es de 109 días.

Hallazgo para la Fase 3: casi todo el costo que paga la estrategia en ese período (~1.2–1.35 pp por activo) es la **compra inicial** de la posición. El benchmark buy & hold no paga esa entrada, lo que genera una asimetría a favor del benchmark que conviene corregir cobrándole la misma compra inicial.

**Corrida real de la Fase 0 (datos hasta 2026-09-25).** Estrategia Sharpe 1.00 vs. benchmark 1.25; MDD −14.65% vs. −15.19%; 112 días. W_i: GGAL 2747, PAMP 2747, YPFD 2721, BMA 2735, BYMA 2161. PAMP y BYMA no cambiaron nunca de exposición. La brecha es casi toda costo de entrada y nivel de exposición; el timing aporta ±1 pp. Sigue con look-ahead: no reportable.

### Fases 1 y 2: entrenamiento point-in-time y ventanas justificadas

| # | Archivo | Problema en la Entrega #3 | Corrección |
|---|---|---|---|
| 1.1 | `signal_generator.py` | El logit se ajustaba **una vez** con toda la muestra: la exposición de cada día usaba eventos futuros (feedback de la Entrega #3). | Walk-forward: para la exposición del día *t* se reajusta el logit con **solo** las filas cuyo horizonte terminó antes de *t* (`known_at < t`). Reajuste diario (`refit_step=1`) con arranque en caliente. Queda un modo `walk_forward=False` solo para medir el efecto del look-ahead. |
| 1.2 | `signal_generator.py` | El evento de la fila *t* usaba r_{t+1..t+h}, pero la exposición de *t* se aplica al retorno de *t*: desalineado un día. | El evento usa r_t..r_{t+h−1}; `known_at` = fecha de r_{t+h−1}. |
| 1.3 | `signal_generator.py` | τ_i de King-Zeng se calculaba con toda la muestra. | τ point-in-time (`excedencias`: frecuencia histórica hasta *t*, mínimo 252 ruedas), o `modelo` (probabilidad de evento implicada por el NIG). |
| 1.4 | `signal_generator.py` | Errores Newey-West sobre un ajuste in-sample. | HAC sobre el **último** ajuste walk-forward, con la misma muestra de ese ajuste. `fit.history` guarda cada reajuste (fecha, n, eventos, α, τ, pendientes). |
| 2.1 | `window_selection.py` | El criterio de W_E (cambio relativo < 10% en α, β, δ, μ) llegaba al techo en los 5 activos. | El problema es que exige estabilidad en parámetros individuales, y μ ≈ 0 hace inestable su cambio relativo aunque la señal no cambie. Criterio configurable: `parametros` (original), `sin_mu`, `estandarizado` (cambio < 1 error estándar vía Hessiano numérico), `estimadores` (cambio relativo < 10% en λ̂, q05, q95, σ). Combinación `suma` (original) o `max`. |
| 2.2 | `window_selection.py` | No se distinguía una estabilización genuina de un candidato que cumple solo al final de la grilla. | `we_status`: `estabiliza` (≥ 3 candidatos seguidos cumplen), `en_borde`, `no_estabiliza`. |
| 2.3 | `nig_model.py`, `levy_estimators.py` | Solo percentiles diarios. | Percentiles del NIG a *h* días (convolución: δh, μh), `prob_outside` y errores estándar. Caché en disco de la calibración (`cache_nig/`), por ticker, W, paso, h y huella de los datos. |
| 2.4 | `main.py` | Script lineal. | `PipelineConfig`, `run_pipeline`, `signal_diagnostics` (tabla "señal viva"), `compare_window_criteria` (solo diagnóstico, sin desempeño) y `save_results`. |

#### Hallazgo: la definición de evento de la especificación

La especificación define el evento como r_{t→t+h} (5 días) fuera de los percentiles 5/95 **diarios** del NIG. Bajo el propio NIG calibrado, eso ocurre ~45% de las veces, no ~10%: un retorno de 5 días tiene ~√5 veces la dispersión de uno diario. El logit entonces no modela "eventos de cola" sino "semanas movidas". Se agregó la variante `event_definition="horizonte"` (percentiles del NIG a 5 días, ~10% de eventos).

#### Decisiones cerradas con la corrida 1 (datos reales, paso 5)

Se eligieron con diagnósticos de cobertura y coherencia, **no** con Sharpe. La especificación original sigue disponible por configuración. Quedan para confirmar con el director.

| Decisión | Especificación original | **Cerrado** | Evidencia de la corrida 1 |
|---|---|---|---|
| Criterio de W_E | `parametros` / `suma` | **`estimadores` / `max`** | Único criterio en que los 5 activos estabilizan y todos cubren ≥ 2 episodios (3,9–7,2 años evaluables). `parametros` no estabiliza en ninguno y deja 2–48 días. |
| Definición de evento | `diaria` | **`horizonte`** | Con `diaria`, 41,6% de las observaciones son "eventos": mide semanas movidas, no colas. Con `horizonte`, 9,94% observado vs. 10% implicado por el NIG: coherente y además valida la calibración. |
| τ para King-Zeng | `excedencias` (frecuencia diaria de \|z\|>2, ~4%) | **`modelo`** (frecuencia del mismo evento, 10%) | τ tiene que ser la frecuencia poblacional del **mismo** evento que se etiqueta. Con `excedencias`, τ ≈ 4% frente a ȳ ≈ 42% bajaba α_i en ~2,8: probabilidad ≈ 5% y exposición fija ≈ 95% con cualquier señal. Con la **misma** señal, la exposición media iba de 95,6% (`excedencias`) a 53,5% (`modelo`): el nivel lo fijaba τ, no λ. |

Nota sobre King-Zeng: la corrección de intercepto por τ está pensada para muestras armadas eligiendo casos por el resultado. Esta muestra es la serie completa, así que con τ coherente la corrección es casi nula (ȳ ≈ τ ≈ 10%). Se mantiene en el código por fidelidad a la especificación y queda documentado que su efecto es menor.

Resultados de la corrida 1 con la configuración anterior (`estimadores`/`max`, `diaria`, `excedencias`), solo como referencia: Sharpe 1,82 vs. 1,83 del benchmark, MDD −43,9% vs. −48,6%, exposición media 95%. Hasta 2022 el portafolio es solo BYMA (W = 350); hay que reportar por período de composición.

#### Parámetros de arranque del walk-forward (supuestos operativos)

| Parámetro | Valor | Motivo |
|---|---:|---|
| `MIN_TRAIN_OBS_PER_ASSET` | 60 | Un activo entra al logit con al menos 60 filas etiquetadas y cerradas. |
| `MIN_TRAIN_EVENTS_PER_ASSET` | 5 | Evita efectos fijos no identificados (separación perfecta). |
| `MIN_TRAIN_EVENTS` | 20 | Mínimo de eventos totales para ajustar. |
| `MIN_HISTORY_FOR_TAU` | 252 | Historia mínima para τ por excedencias. |

Antes de que se cumplan, el activo no tiene exposición decidida por el modelo (no entra al backtest).

#### Validación de las Fases 1-2

- Eventos cerrados: en cada reajuste, el tamaño de la muestra coincide exactamente con las filas con `known_at < fecha de ajuste` (el historial queda en `logit_history.csv`).
- Los criterios de W_E se compararon sobre los datos reales (`window_criteria_comparison.csv`), sin mirar desempeño.

---

## Elecciones de implementación vigentes

### 1. Grilla de búsqueda `{250, ..., 2500}` (Sección 2)

La especificación no fija el paso de la grilla. Se usa paso de 50 ruedas (`GRID_STEP` en `window_selection.py`): 46 puntos entre 250 y 2500. Un paso más fino cambia el resultado en, a lo sumo, 50 días, pero multiplica el costo computacional (cada punto de la grilla implica un MLE completo de NIG para `W_E,i`).

### 2. Criterio de "cobertura" para `W_i^Kaufman` (Sección 2)

Es el punto más abierto de la especificación: pide "la menor W candidata que cubre ≥4 de los 5 episodios del catálogo" sin decir qué significa "cubrir". Se adoptó: un episodio está cubierto por la ventana `W` si, calculando una intensidad de saltos empírica (excedencias `|z|>2`) sobre ventana rodante `W`, el z-score de esa intensidad supera `1.0` en alguna rueda dentro de ±5 ruedas de alguna fecha del episodio. Es una interpretación razonable pero **no la única posible**, y hay que confirmarla con el director porque determina directamente `W_i` para los 5 activos.

### 3. Estandarización `z_lambda` / `z_dlambda` con historia insuficiente (Sección 3)

Una lectura estricta de "estandarizada dentro de `W_i`" implicaría no tener ningún z-score hasta `2×W_i` ruedas de historia. Con los W_i de la Entrega #3 (~2700 ruedas) eso vaciaba el panel del logit. Se usa un piso **fijo** de 30 observaciones (`Z_SCORE_MIN_PERIODS` en `levy_estimators.py`) en el rolling de estandarización. No hay look-ahead (solo se usan datos pasados), pero los primeros z-scores son menos estables. Con los criterios de W de la Fase 2 las ventanas pueden ser más cortas; si W_i baja lo suficiente, se puede volver a una lectura más estricta.

### 4. Errores estándar Newey-West sobre el panel pooled (Sección 4)

Los modelos discretos de `statsmodels` (Logit) no exponen `get_robustcov_results` en la versión usada. Se implementó a mano un sandwich `Bread × Meat × Bread` con ponderación de Bartlett a 4 rezagos, con `Meat` acumulado **dentro de cada activo** (Newey-West clustered por activo, sin correlación contemporánea entre activos tipo Driscoll-Kraay). Si el director pide Driscoll-Kraay, hay que extender `_newey_west_panel_cov` en `signal_generator.py`.

### 5. Combinación equal-weight con historias desparejas (Sección 5)

BYMA tiene historia más corta (desde 2017). El overlay equal-weight (`run_multi_asset_backtest`) promedia, cada día, los `net_return` de los activos que **sí** tienen dato ese día. Promediar retornos diarios equivale a **rebalancear a pesos iguales todos los días**; el costo de ese rebalanceo entre activos todavía no se cobra. El benchmark (`equal_weight_benchmark`) hace lo mismo y además **sin costos**, como señaló el feedback de la Entrega #3. Se resuelve en la Fase 3.

### 6. Parámetros todavía abiertos

- `risk_free_rate`: **cerrado** (ver sección «Variables macro» al final). Se usa la tasa de pases entre terceros a 1 día del BCRA (proxy de la caución); el default 0.0 de `performance_metrics.py` queda solo por compatibilidad.
- El costo de transacción **ya está cerrado** (ver tabla de la Fase 0).
- Excepción BYMA de W^Kaufman: la especificación relaja el objetivo a "6 eventos", pero el catálogo tiene 5 episodios (18 fechas). Se mantiene el valor de la especificación y se deja anotado para confirmar con el director.
- `2024-07-24` quedó fuera del catálogo (Fase 0).

### 7. Qué falta antes de reportar resultados

1. ~~Corrida 1~~: hecha. Criterio de W, evento y τ cerrados (tabla de arriba).
2. Justificar o acotar el W = 350 de BYMA (los demás activos están en 1.650–1.800), porque define la composición del portafolio hasta 2022.
3. Decidir el tratamiento del período 2024+: congelar la especificación antes de mirarlo (holdout) o declararlo usado.
4. Fases 3-5 (comparador, "¿protege?", costos diarios y liquidez) antes de la corrida final con `RECALIBRATION_STEP=1`.

---

## Corrida con la configuración cerrada (`estimadores`/`max`, `horizonte`, `modelo`; paso 5)

- La tasa de eventos del último ajuste es 10,0%, las constantes King-Zeng casi no cambian (τ = ȳ) y la exposición ahora se mueve: 48 cambios, con mínimos de entre 54% y 94% según el activo.
- Estrategia vs. benchmark: Sharpe 2,093 vs. 2,094; MDD −38,4% vs. −41,3%; retorno 117,7% vs. 132,3%; volatilidad 41,2% vs. 45,1%.
- **Timing ≈ 0.** Una exposición **fija** igual a la media de cada activo da prácticamente el mismo resultado (retorno 116,3%, volatilidad 41,1%, MDD −38,0%), también por período de composición. La mejora en drawdown se explica por estar menos invertido, no por el momento en que se baja la exposición. Esa exposición fija es el comparador que corresponde en la Fase 4.
- El evento es de dos colas: en GGAL y YPFD, con exposición baja aumentan tanto las caídas como las subas grandes. Consulta 10 en `CONSULTAS_DIRECTOR.md`.
- `estandarizado` no es reproducible entre corridas (depende del Hessiano numérico); `estimadores` sí.

---

## Fase 3: comparador

| # | Archivo | Problema en la Entrega #3 | Corrección |
|---|---|---|---|
| 3.1 | `comparators.py` (nuevo) | La cartera promediaba cada día los retornos de los activos: rebalanceo diario a pesos iguales **gratis**, para la estrategia y para el benchmark. | Motor de tenencias (`simulate_portfolio`): cada activo es un bolsillo con posición y caja, las posiciones derivan con los precios, se rebalancea entre activos con una frecuencia **explícita** (`diario`, `mensual`, `trimestral`, `nunca`) y cuando entra un activo, y dentro de un activo solo se opera cuando cambia la exposición. Toda operación paga el costo por lado; el tamaño de la posición descuenta el costo, así que nunca queda caja negativa. |
| 3.2 | idem | El benchmark no pagaba la compra inicial. | Estrategia y comparadores pasan por el mismo motor: todos pagan la compra inicial. |
| 3.3 | idem | Un único comparador pasivo. | EW 100%, exposición fija (media ex post; es un diagnóstico, no una regla implementable), vol targeting (σ* / σ̂ con σ̂ de 60 ruedas, σ* = mediana expanding, ambos hasta t−1, histéresis de 5 p.p.), MA200 (precio de t−1 contra su media de 200 ruedas) y Merval (índice, sin costos). Todos con los mismos activos y fechas que la estrategia. |
| 3.4 | `main.py` | — | `PipelineConfig.rebalance` (default `mensual`). `result["comparison"]` sale del motor nuevo; `result["comparison_e3"]` conserva la versión anterior como referencia. Nuevos CSV: `performance_by_period.csv` (por tramo de composición), `portfolios_nav.csv` y `performance_comparison_e3_reference.csv`. |

**Anticipo con tus datos de la corrida anterior** (mismas exposiciones, calculado a partir de los CSV por activo; sin vol targeting, MA200 ni Merval, porque requieren la historia previa):

| Rebalanceo | Cartera | Retorno anual | Sharpe | MDD | Costo total (pp) |
|---|---|---:|---:|---:|---:|
| E3: promedio diario sin costo | Estrategia | 117,7% | 2,093 | −38,4% | — |
| E3: promedio diario sin costo | EW 100% | 132,3% | 2,094 | −41,3% | — |
| diario con costos | Estrategia | 112,1% | 2,030 | −39,7% | 22,6 |
| diario con costos | EW 100% | 125,1% | 2,026 | −42,5% | 21,7 |
| **mensual** | **Estrategia** | 117,8% | 2,083 | −38,7% | 9,9 |
| **mensual** | **EW 100%** | 130,8% | 2,074 | −41,5% | 8,4 |
| mensual | Exposición fija | 116,6% | 2,070 | −38,3% | 8,0 |
| nunca (buy & hold) | Estrategia | 117,9% | 2,004 | −44,9% | 6,0 |
| nunca (buy & hold) | EW 100% | 129,0% | 2,020 | −45,5% | 4,2 |

Lectura: el supuesto de rebalanceo mueve el Sharpe en ±0,08 y el drawdown en hasta 6 p.p., pero afecta de forma parecida a la estrategia y al benchmark. La conclusión de la Fase 2 no cambia: la estrategia ≈ exposición fija.

**Caja al 0% (corridas anteriores a las variables macro).** En pesos, la caja al 0% pierde contra la inflación. Además, con rebalanceo `nunca` la exposición de la estrategia sube sola (de 91% a 95% en promedio) porque las acciones suben en pesos y la caja no. Esto penaliza a cualquier estrategia que tenga caja; se resolvió con la tasa de pases a 1 día del BCRA como proxy de la caución (consulta 12; sección «Variables macro»).

**Corrida real de la Fase 3** (`estimadores`/`max`, `horizonte`, `modelo`, rebalanceo mensual, paso 5; 1.742 días):

| | Estrategia | EW 100% | Exp. fija | Vol targeting | MA200 | Merval |
|---|---:|---:|---:|---:|---:|---:|
| Retorno anual | 117,7% | 130,7% | 116,2% | 99,8% | 87,6% | 96,5% |
| Sharpe | 2,086 | 2,074 | 2,070 | 1,996 | 1,782 | 1,732 |
| Max Drawdown | −38,5% | −41,5% | −38,2% | −39,5% | −45,2% | −48,7% |
| Calmar | 3,05 | 3,15 | 3,04 | 2,53 | 1,94 | 1,98 |
| Costo total (pp) | 9,9 | 8,3 | 7,9 | 12,8 | 48,0 | 0 |

- La estrategia supera a vol targeting y a MA200 en todas las métricas. MA200 queda muy penalizado por los costos (turnover anual de 5,7).
- Frente al EW 100%: igual Sharpe, 2,9 p.p. menos de drawdown y 13 p.p. anuales menos de retorno. Calmar algo menor.
- Frente a la exposición fija: prácticamente igual en total y en el tramo de 5 activos (Sharpe 2,003 contra 2,006). Solo en el tramo en que la cartera es únicamente BYMA (2019–2022, W = 350) la estrategia supera a la exposición fija (Sharpe 1,92 contra 1,86).
- El rebalanceo mensual queda muy cerca de la referencia de la Entrega #3. El rebalanceo diario con costos es el peor para las dos carteras.

---

## Fase 4: ¿protege?

Módulo nuevo `protection.py`. Parámetros adoptados: episodio = ±10 ruedas hábiles de cada fecha del catálogo; régimen = vol de 60 ruedas del EW contra su mediana (ex post, solo para atribución); VaR/ES rolling de 252 ruedas al 95% y 99%.

| Bloque | Función | Qué responde |
|---|---|---|
| Segmentación | `segmented_metrics` | Métricas dentro/fuera de episodio y en vol alta/baja, por cartera |
| Por episodio | `per_episode_table` | Retorno acumulado y drawdown en cada episodio cubierto |
| Leave-one-episode-out | `leave_one_episode_out` | ¿La ventaja dentro de episodios depende de uno solo? |
| Riesgo de cola | `tail_risk_table` | VaR/ES out-of-sample, Kupiec, Christoffersen (π11/π01), cobertura condicional |
| Bootstrap | `bootstrap_table` | IC 95% de la diferencia de Sharpe, retorno, drawdown y Calmar; bootstrap estacionario pareado, bloque de 20, 2.000 réplicas |
| Sobreajuste | `run_configuration_grid`, `overfitting_summary` | Deflated Sharpe (grilla de 32 configuraciones definidas; N = 30 con resultado) y PBO por CSCV (16 bloques, 12.870 combinaciones) |

**Anticipo con datos reales** (calculado sobre `portfolios_nav.csv` de la corrida de la Fase 3; paso 5):

Bootstrap, estrategia − comparador:

| Comparador | Sharpe | Retorno anual | Max Drawdown |
|---|---|---|---|
| EW 100% | +0,012 [−0,027; +0,060], p = 0,29 | −13,0 pp [−28,2; −4,2] | **+2,9 pp [+1,1; +5,5], p = 0,000** |
| Exposición fija | +0,016 [−0,021; +0,066], p = 0,24 | +1,5 pp [−2,0; +5,5], p = 0,21 | −0,3 pp [−1,3; +2,8], p = 0,67 |
| Vol targeting | +0,090 [−0,027; +0,225], p = 0,08 | **+17,8 pp [+3,6; +40,3], p = 0,004** | +1,0 pp [−5,1; +5,5], p = 0,44 |

- **Protege frente al EW:** menor drawdown, de forma significativa. Pero es lo que da **cualquier** reducción de exposición: contra la exposición fija, ninguna diferencia es distinguible del azar. No hay evidencia de que el timing aporte.
- **Contra vol targeting:** mismo drawdown con 18 pp más de retorno anual (significativo). Es una defensa más barata que la regla por volatilidad.
- **Dentro de episodios:** Sharpe 1,39 (estrategia) contra 1,34 (EW) y 1,34 (fija); peor día −14,5% contra −15,7% (EW). En leave-one-out, la ventaja sobre la exposición fija se mantiene salvo sin PASO 2019 (1,32 contra 1,35): depende de ese episodio, donde la estrategia sube 10,1% contra 6,4% de la fija y cae 17,8% contra 22,0%.
- **Riesgo de cola:** VaR 95% medio de 3,64% contra 3,95% (EW) y ES 95% de 4,88% contra 5,31%, iguales a los de la exposición fija (3,61% y 4,85%). Kupiec no rechaza al 95%. Al 99%, en la estrategia Christoffersen rechaza independencia (p = 0,033, π11/π01 = 7,4) y en el EW no (p = 0,059, 5,5): la estrategia **no** reduce el agrupamiento de excepciones extremas. 

**Deflated Sharpe y PBO con la grilla real** (30 configuraciones con resultado; `config_grid_returns.csv`):

| Historia mínima | Configs | Fechas comunes | Sharpe elegida | SR0 (azar) | DSR | PBO |
|---|---:|---|---:|---:|---:|---:|
| 1 año | 26 | 2025-09 → 2026-09 | 1,16 | 0,21 | 0,85 | 0,56 |
| **2 años (default)** | 20 | 2023-11 → 2026-09 | 1,48 | 0,20 | **0,985** | **0,34** |
| 3 años | 16 | 2023-04 → 2026-09 | 1,92 | 0,14 | 1,00 | 0,61 |

- **DSR ≈ 0,99:** el Sharpe de la configuración elegida no se explica por haber elegido la mejor entre 30. Pero ese Sharpe es básicamente el del mercado: el EW tiene el mismo. El DSR valida el Sharpe **absoluto**, no una ventaja sobre el benchmark.
- **PBO entre 0,34 y 0,61:** la configuración que mejor rinde in-sample no se sostiene out-of-sample más de lo que daría el azar, y la pendiente IS→OOS es ≈ −1. Es coherente con la Fase 4: las configuraciones difieren en ruido, no en una capacidad de anticipar distinta.
- **El DSR ingenuo** (Sharpes de períodos distintos) da 0,36: es un artefacto. La dispersión de Sharpes entre configuraciones con 48 días y con 7 años infla SR0 a 2,2. Por eso la función usa el período común.
- **La configuración elegida** (`estimadores/max|horizonte|modelo`) queda 5.ª de 30 por Sharpe en su propio período. No se eligió por Sharpe, sino por cobertura (Fase 2).

---

## Fase 5: liquidez, costo variable y regla de salida

Módulo nuevo `liquidity.py`. `simulate_portfolio` ahora acepta el costo como matriz fecha × activo y puede devolver el monto operado por día.

| # | Problema detectado | Implementación |
|---|---|---|
| 5.1 | Costo con la media de Corwin-Schultz; la serie diaria de spread `spread_cs` no se usaba. | `cost_mode="variable"` (nuevo caso base): costo por lado = 66,55 pb + medio spread CS diario (High/Low sin ajustar, variante backward-looking, truncado en 0). El spread se suaviza con la media de 20 ruedas y se usa el valor conocido al cierre de t−1, que es cuando se ejecuta la orden. `medio` y `p95` quedan como escenarios de sensibilidad (`cost_sensitivity`). |
| 5.2 | Tope de 10% del ADV sin implementar. | `adv_diagnostics`: ADV_20 en pesos (Close × Volume) conocido en t−1. Cada orden se expresa como fracción de la cartera del día y se compara con un AUM de ARS 500 M, interpretado como valor de la cartera **en cada fecha**. Informa violaciones por activo y por año, y la **capacidad**: valor máximo de cartera, en pesos de la fecha de la orden, con el que el 95% de las órdenes respeta el tope. Es diagnóstico: no se fuerza en el backtest. |
| 5.3 | Regla de salida sin implementar. | `apply_drawdown_floor`: techo = max(E_min, 100% − κ(\|DD\| − D_max)) cuando el precio cae más de D_max desde su máximo de 252 ruedas (conocido en t−1). Exposición = min(señal, techo), con la misma banda de 5 p.p. para no operar a diario. Desactivado por defecto (`drawdown_floor=None`); el notebook lo compara con D_max = 20%, κ = 2, E_min = 25%, y la grilla de parámetros va en la Fase 6. |

**Supuesto a declarar:** el AUM de 500 M de ARS es nominal. Con la inflación del período, en 2019 era una cartera grande frente al ADV de BYMA y en 2026 es chica. Por eso el resultado a reportar es la capacidad por año, no el conteo de violaciones con un AUM fijo.

**Corrección tras la primera corrida real:** la primera versión expresaba las órdenes en unidades del capital **inicial**. Con un NAV que se multiplicó unas 200 veces en pesos, eso equivale a suponer 500 M de ARS invertidos en ago-2019 que después crecen con la cartera. Así se exageraban las violaciones (35%–85% de las órdenes) y la capacidad quedaba en pesos de 2019. Ahora cada orden es fracción de la cartera del día y la capacidad está en pesos de esa fecha.

**Corrida real de la Fase 5** (costo variable como caso base; 1.743 días):

- **Costo variable ≈ costo medio.** La estrategia paga 10,2 pp contra 9,9 pp y el Sharpe es 2,078 contra 2,079. Con costo p95 el Sharpe baja a 2,042 y el EW a 2,038. El orden estrategia ≥ EW ≥ exposición fija en Sharpe se mantiene con los tres costos. Con un turnover de ~1,2 veces por año, el costo no define la conclusión.
- **Validación del spread** (±5 ruedas del episodio contra la media del año):
  - PASO 2019: 1,7–2,1 veces en los 5 activos.
  - Ciclo electoral 2023: 1,3–1,9 veces.
  - Era Milei: 1,2–1,5 veces.
  - Covid y crisis 2018: mixto (0,75–1,8 veces).
- **Piso por drawdown (D_max = 20%, κ = 2, E_min = 25%): empeora.**
  - Sharpe 1,997 contra 2,078. La diferencia es significativa: IC 95% [−0,160; −0,011].
  - Retorno anual −15,1 pp y costos 15,9 pp contra 10,2 pp.
  - Drawdown solo 0,8 p.p. menor, no significativo.

  El recorte llega tarde y deja afuera la recuperación. **Decisión documentada: no se adopta.** La Fase 6 prueba otros parámetros para verificar que no depende de esta elección.
- Los archivos `adv_*.csv` de esta corrida son anteriores a la corrección de la capacidad: hay que regenerarlos.

---

## Fase 6: sensibilidad a los supuestos

Módulo nuevo `sensitivity.py`. `levy_estimators.py` ahora deriva λ (con otro u) y los percentiles a h días (con otro h) a partir de los parámetros NIG ya calibrados, sin repetir el MLE. `PipelineConfig.u_multiplier` es nuevo, y los rezagos de Newey-West siguen a h (h − 1).

| Bloque | Función | Qué varía |
|---|---|---|
| Un parámetro por vez | `run_ofat` | Señal: u ∈ {1,5; 1,75; 2; 2,5; 3}, h ∈ {3; 5; 7; 10}, criterio de W (3 alternativas). Cartera: banda ∈ {0; 2; 5; 8; 12} p.p., costo (solo fijo, medio, variable, p95, 350 pb), rebalanceo |
| Tornado | `tornado_table` | Rango por parámetro, sobre el Sharpe y sobre Sharpe − exposición fija |
| Factorial | `factorial_two` | costo × horizonte (el criterio de W queda fuera del factorial) |
| Piso por drawdown | `drawdown_floor_grid` | D_max ∈ {10; 20; 30}%, κ ∈ {1; 2; 4}, E_min ∈ {25; 50}% |
| Por activo | `by_asset_table` | Cada activo contra su buy & hold y contra su exposición fija |

Cada variante reporta la diferencia contra el EW (¿protege?) y contra la exposición fija a su propia media (¿aporta el timing?). Todas las variantes, incluida la base, reajustan el logit cada 5 ruedas.

**Anticipo con datos reales (solo parámetros de cartera**, calculados sobre los CSV de la corrida de la Fase 5):

| Variante | Sharpe | Sharpe − EW | MDD − EW | Sharpe − exp. fija |
|---|---:|---:|---:|---:|
| base | 2,079 | +0,012 | +2,9 pp | +0,016 |
| banda 0 / 2 / 8 / 12 p.p. | 2,061–2,076 | −0,006 a +0,009 | +2,9 a +3,5 pp | −0,005 a +0,016 |
| costo solo fijo / p95 / 350 pb | 2,015–2,094 | −0,002 a +0,016 | +2,9 pp | +0,003 a +0,019 |
| rebalanceo diario / trimestral / nunca | 2,001–2,075 | −0,010 a +0,010 | +0,8 a +3,1 pp | +0,016 a +0,018 |

- **Protección frente al EW:** robusta. El MDD es menor en todas las variantes (entre +0,8 pp con rebalanceo "nunca" y +3,5 pp).
- **Timing:** Sharpe − exposición fija entre −0,005 y +0,019 en todas las variantes. La conclusión "protege por nivel, no por timing" no depende de la banda, del costo ni del rebalanceo.
- **Por activo:** la ventaja de timing tiene signo mixto (GGAL −0,020; YPFD −0,013; BMA +0,017; PAMP +0,013; BYMA +0,005), así que ningún activo sostiene un timing sistemático. El MDD contra el buy & hold es menor en los cinco activos.
- **Piso por drawdown:** ninguna de las 18 combinaciones supera el Sharpe de la estrategia sin piso. Las agresivas (κ = 4, D_max = 10%) pagan hasta 46 pp de costos. Confirma que no se adopta. Este anticipo usa solo la historia desde el inicio de cada activo, así que el drawdown de 252 ruedas es aproximado al comienzo.
- Faltan los parámetros de señal (u, h, criterio de W): salen con la corrida del notebook.

---

## Variables macro, ventanas fijas y reproducibilidad (Entrega #4, cierre)

### Variables macro de control (`macro_data.py`)

Prometidas en la Entrega #2 como «datos de control/contexto». **No son insumo de la señal**: la señal depende solo de precios y por eso la caché del NIG no cambia.

| Uso | Serie | Convención |
|---|---|---|
| Tasa libre de riesgo y rendimiento de la caja, en pesos | Pase entre terceros a 1 día del BCRA (id 150; proxy de la caución a 1 día), TNA % | Base 365; tasa vigente al cierre de t−1; devenga por días corridos (fines de semana y feriados incluidos); antes de la primera observación, 0% |
| Conversión a dólares | USD/ARS oficial (BCRA) | r_usd = (1 + r_ars) / (1 + ΔFX) − 1; la señal y las exposiciones no se recalculan |
| Caja en dólares | T-bill 3m EE.UU. (`^IRX`) | Base 360, tasa de descuento |

- `simulate_portfolio(..., cash_rate=...)`: la caja de cada bolsillo devenga la tasa diaria. `summary`/`compare` y el bootstrap, DSR y PBO usan retornos en exceso de la misma serie. Calmar no se ajusta.
- Cada serie se lee de un snapshot CSV en `datos/macro/` (`fecha,valor`); si no existe, se descarga y se guarda. El BCRA no publica la caución bursátil, así que se usa el id 150 ("Tasa de interés por operaciones de pases entre terceros a 1 día", diaria, desde 2008; 2.856 observaciones, TNA %, con el salto de agosto 2019 visible). La API vigente es v4.0 (la v3.0 devuelve 410 Gone). Si la descarga falla, basta dejar el CSV a mano.
- **Advertencia:** las descargas de la tasa (monetarias v4.0, id 150) y del USD/ARS oficial (cotizaciones cambiarias v1.0) fueron verificadas contra la API real por el autor (2.856 y 2.864 observaciones, hasta sep-2026); la API monetaria v3.0 ya no existe (410 Gone). El T-bill sale de `yfinance` (`^IRX`).
- **Limitación:** con el cepo (oct-2019 a abr-2025) el oficial subestima la depreciación efectiva; la vista en USD es una cota.

### Robustez con ventana corta y fija (`sensitivity.run_fixed_window_grid`)

Respuesta al feedback de la Entrega #3 («más historia utilizable»). La misma W ∈ {500, 750, 1000} para los 5 activos, contra la especificación principal (`estimadores`/`max`). **No cambia la especificación principal ni son candidatas a elegir**: no entran a la grilla de DSR/PBO. Estas W no pasan el criterio de estabilización de W_E, así que el NIG es más ruidoso. Cada fila reporta cobertura (inicio, inicio con los 5 activos, años y episodios cubiertos), métricas en el período propio y en el período común, y un bootstrap pareado de Sharpe(estrategia) − Sharpe(exposición fija) con veredicto sobre el timing. BYMA (listado en 2017) empieza más tarde que el resto aun con W = 500.

### Vista en dólares (`sensitivity.usd_view`)

Las mismas carteras (estrategia, EW, exposición fija, vol targeting, MA200), con las mismas exposiciones y costos, en pesos y en dólares, más bootstrap de las diferencias contra EW y exposición fija en cada moneda.

### Corrida final (paso 1, con caja y Sharpe en exceso de la tasa del pase a 1 día)

Período 2019-08-13 a 2026-09-29 (1.744 ruedas); rebalanceo mensual, costo variable (Corwin-Schultz); exposición media 90,3%.

| Cartera | Retorno anual | Sharpe | Max Drawdown | Costo total (pp) |
|---|---|---|---|---|
| Estrategia | 128,5% | 1,123 | −36,1% | 12,1 |
| EW 100% | 135,7% | 1,120 | −41,5% | 8,2 |
| Exposición fija (media ex post) | 128,2% | 1,119 | −36,0% | 7,8 |
| Vol targeting | 118,0% | 1,043 | −38,2% | — |
| MA200 | 110,1% | 0,927 | −40,5% | — |

- **Protege por nivel, no por timing.** Contra EW: Sharpe +0,003 [−0,051; +0,050] (no distinguible) y MDD +5,4 pp [+1,3; +7,9], p = 0,005. Contra la exposición fija: Sharpe +0,004 [−0,051; +0,054], MDD −0,1 pp [−2,9; +2,9]. Contra vol targeting: Sharpe +0,080 [−0,050; +0,219] y retorno +10,5 pp (p = 0,049, pero con IC95 [−1,8; +28,0] que incluye el cero; con 12 pruebas y sin corrección por comparaciones múltiples no se presenta como ventaja firme), sin diferencia significativa en MDD.
- **Segmentos:** dentro de episodios Sharpe 0,64 contra 0,58 (EW) y 0,59 (fija); fuera, 1,28 contra 1,29 y 1,29. El resultado dentro de episodios depende de `ciclo_electoral_2023`: sin ese episodio el Sharpe cae a 0,18 contra 0,07 (EW). Peor día −14,1% contra −15,7%.
- **Cola:** VaR 95% medio 3,54% contra 3,94% (EW); ES 95% 4,78% contra 5,31%; idénticos a la exposición fija. Kupiec no rechaza al 95% (p = 0,96). Al 99% Christoffersen rechaza independencia (p = 0,004, π11/π01 = 9,6), igual que EW (p = 0,006): la estrategia no reduce el agrupamiento de excepciones extremas.
- **DSR y PBO** (grilla de 30 configuraciones, T = 721 ruedas comunes): Sharpe de la elegida 0,69, SR0 0,13, DSR = 0,83; PBO = 0,35. Con DSR = 0,83 el Sharpe queda por debajo del umbral convencional de 0,95: hay evidencia moderada, no concluyente, de que no se explica por la selección entre configuraciones. Y en todo caso es el Sharpe absoluto, no una ventaja sobre el benchmark.
- **Sensibilidad del DSR y el PBO a la historia mínima exigida** (`fase4_dsr_pbo_sensibilidad.csv`):

| Historia mínima | Configs | Fechas comunes | Sharpe elegida | SR0 | DSR | PBO |
|---|---|---|---|---|---|---|
| 252 días | 26 | 2025-09-01 → 2026-09-29 | 0,416 | 0,129 | 0,619 | 0,716 |
| 504 días (adoptada) | 20 | 2023-10-25 → 2026-09-29 | 0,688 | 0,129 | 0,832 | 0,349 |
| 756 días | 12 | 2022-12-26 → 2026-09-29 | 0,913 | 0,130 | 0,936 | 0,476 |

  El DSR va de 0,62 a 0,94 y nunca supera 0,95; el PBO va de 0,35 a 0,72 (con 252 días queda por encima de 0,5). La conclusión depende de la historia mínima exigida, que es un supuesto: no se presenta el resultado como validación de la selección. Por Sharpe en su propio período (refit cada 5 ruedas), la configuración elegida (`estimadores/max|horizonte|modelo`, 1,118) es la primera de las 30. Se eligió por cobertura y no por Sharpe (Fase 2), pero esa coincidencia es justamente la situación que el DSR y el PBO intentan penalizar, y por eso conviene reportarla.
- **Sensibilidad un parámetro a la vez** (`fase6_ofat_macro.csv`; la fila base da Sharpe 1,118 porque el logit se reajusta cada 5 ruedas):

| Variante | Sharpe | Sharpe − EW | MDD − EW | Sharpe − exp. fija |
|---|---|---|---|---|
| base | 1,118 | −0,001 | +5,8 pp | +0,000 |
| u = 1,5 / 1,75 / 2,5 / 3 | 1,108–1,143 | −0,011 a +0,024 | +5,0 a +5,8 pp | −0,010 a +0,024 |
| h = 3 / 7 / 10 | 1,025–1,108 | −0,017 a +0,026 | +4,5 a +5,6 pp | −0,018 a +0,027 |
| criterio de W: estimadores/suma · estandarizado/max · sin_mu/max | 0,808 · 0,963 · 0,695 | −0,050 · −0,006 · −0,119 | +5,0 · +5,5 · +4,7 pp | **−0,056 · −0,012 · −0,122** |
| banda 0 / 2 / 8 / 12 p.p. | 1,092–1,132 | −0,027 a +0,012 | +4,6 a +5,9 pp | −0,026 a +0,014 |
| costo solo fijo / medio / p95 / 350 pb | 1,044–1,139 | −0,029 a +0,007 | +5,8 pp | −0,025 a +0,008 |
| rebalanceo diario / trimestral / nunca | 1,055–1,126 | −0,028 a −0,006 | +3,4 a +5,8 pp | −0,002 a +0,003 |

  - **Protección frente al EW: robusta.** El MDD es menor en todas las variantes (de +3,4 pp con rebalanceo «nunca» a +5,9 pp).
  - **Timing: nunca aporta de forma clara.** Sharpe − exposición fija va de −0,026 a +0,027 en señal, banda, costo y rebalanceo, y no es positivo de forma sistemática. Con criterios de W distintos del elegido el timing **resta** (−0,056 con estimadores/suma; −0,122 con sin_mu/max), porque esas ventanas dan una señal peor. Por eso no se afirma que la columna quede cerca de 0 en *todas* las variantes: queda cerca de 0 salvo en los criterios de W alternativos, donde es negativa.
- **Costos:** el Sharpe casi no cambia con costo medio, variable o p95 (1,124 / 1,123 / 1,079); con p95 los costos son 24,6 pp contra 16,9 pp del EW. Piso por drawdown (D_max 20%, κ 2, E_min 25%): Sharpe 1,036, diferencia −0,087 [−0,171; −0,022]; no se adopta.
- **Liquidez** (AUM supuesto ARS 500 M, tope 10% del ADV_20): GGAL, PAMP, YPFD y BMA con 1,7–5,7% de órdenes sobre el tope. BYMA: 33% de las órdenes, concentradas en 2019–2022 (37–78% por año); desde 2024 sin violaciones. Los resultados de BYMA antes de 2023 no son implementables a esa escala.
- **Ventanas fijas (período propio / período común desde 2019-08-13):**

| Variante | Inicio | Años evaluables | Sharpe | MDD | Sharpe − EW | Sharpe − fija [IC95] | Veredicto |
|---|---|---|---|---|---|---|---|
| Base (`estimadores`/`max`) | 2019-08-13 | 6,92 | 1,118 | −35,7% | −0,001 | +0,000 [−0,049; 0,043] | no distinguible |
| W = 500 | 2017-09-29 | 8,71 | 0,478 | −50,4% | −0,026 | −0,033 [−0,065; −0,002] | **timing resta** |
| W = 750 | 2018-06-26 | 8,02 | 0,525 | −51,0% | +0,003 | −0,003 [−0,025; 0,020] | no distinguible |
| W = 1000 | 2019-07-03 | 7,03 | 0,581 | −50,8% | +0,017 | +0,015 [−0,043; 0,079] | no distinguible |

  En el período común, Sharpe 0,77 / 0,74 / 0,76 (W = 500 / 750 / 1000) contra 1,118 de la base, con MDD entre −42% y −48% contra −36%. Con ventana corta la protección frente al EW se mantiene (+3 a +11 pp de MDD) y el timing no mejora: con W = 500 es significativamente negativo. Los Sharpe más bajos reflejan sobre todo períodos más largos (incluyen 2018–2019, previos a la base) y un NIG más ruidoso; no son candidatas a reemplazar la especificación principal.
- **Vista en dólares** (oficial; cota por el cepo):

| Cartera | Retorno anual USD | Sharpe USD | MDD USD | Sharpe pesos | MDD pesos |
|---|---|---|---|---|---|
| EW 100% | 45,1% | 0,942 | −59,6% | 1,120 | −41,5% |
| Estrategia | 42,4% | 0,945 | −54,4% | 1,123 | −36,1% |
| Exposición fija | 41,9% | 0,936 | −54,4% | 1,119 | −36,0% |
| Vol targeting | 40,1% | 0,946 | −55,9% | 1,043 | −38,2% |
| MA200 | 29,5% | 0,757 | −59,6% | 0,927 | −40,5% |

  Las diferencias de la estrategia frente a EW y exposición fija se mantienen en dólares: Sharpe +0,003 y +0,009 (no significativos), MDD +5,2 pp [+1,7; +8,0] contra EW y 0,0 pp contra la fija. El drawdown en dólares es ~18 pp peor que en pesos para todas las carteras.

### Reproducibilidad

- `datos/precios.pkl`: snapshot de precios (y Merval). Se guardó con tipos respaldados por `pyarrow`, por lo que **hace falta tener `pyarrow` instalado** (está en `requirements.txt`) para leerlo. La caché del NIG se identifica con una huella de los datos que incluye la última fecha: si se vuelve a descargar de Yahoo otro día, la huella cambia y se recalcula todo. Con el snapshot, las corridas siguientes no dependen de la red ni de la fecha.
- `cache_nig/`: calibraciones NIG por (ticker, W, `recalibration_step`, huella). Mantener `RECALIBRATION_STEP = 1`: el paso forma parte del nombre del archivo y cambiarlo invalida la caché.
- Un fallo de disco al leer o guardar la caché o los snapshots avisa pero no corta la corrida.
- Sin caché, la calibración completa toma del orden de 10 horas (≈ 11.000 ajustes por ventana fija y paso 1); con `cache_nig/` y `datos/` incluidos, el notebook corre en minutos. Entregar ambas carpetas junto con el código.
- No incluir `__pycache__`.
