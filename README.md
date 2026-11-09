# 🎯 LeadScore: predicción de conversión de leads

Modelo predictivo que clasifica los leads de una empresa de cursos en línea según su probabilidad de conversión. Está desplegado en una app web con **ranking de leads**, **sistema de priorización** y **dashboard comercial**.

**App en vivo:** https://lead-scoring-app-david-daniers-mora.streamlit.app/

| | |
|---|---|
| **Proyecto** | Proyecto final de la materia Inteligencia de Negocios y Minería de Datos |
| **Institución** | Universidad Cooperativa de Colombia, Facultad de Ingeniería de Software |
| **Profesor** | PhD Cristian Camilo Ordoñez Quintero |
| **Desarrollado por** | David Fernando Ramírez de la Parra · Daniers Alexander Solarte Limas · Juan Felipe Mora Revelo |
| **Dataset** | [Lead Scoring Dataset (Kaggle)](https://www.kaggle.com/code/kratikulhara/lead-scoring-case-study): 9 240 leads × 37 variables |

---

## 1. Problema de negocio

*X Education* vende cursos en línea y recibe miles de **leads** (personas interesadas) al mes, pero solo el **38.5 %** termina comprando. Si el equipo de ventas llama a todos por igual, pierde tiempo con leads que no van a convertir.

**Objetivo general:** desarrollar un modelo predictivo que clasifique los leads según su probabilidad de conversión.

**Objetivo SMART:** asignar a cada lead un score de 0 a 100 % de probabilidad de conversión, con un **ROC-AUC ≥ 0.85** en datos de prueba, desplegado en una app al cierre del semestre.

**Tipo de problema:** aprendizaje **supervisado** de **clasificación binaria**. La variable objetivo es `Converted` (1 = compró, 0 = no compró).

## 2. Requisitos y cómo se cumplen

| Requisito | Cómo se cumple |
|---|---|
| Modelos de clasificación | Regresión Logística (base) y Random Forest |
| Feature engineering | Limpieza, eliminación de fuga de datos, variables nuevas, agrupación de categorías raras y one-hot encoding |
| Gradient Boosting y XGBoost | Ambos entrenados y ajustados con `RandomizedSearchCV`; **XGBoost es el modelo final** |
| Evaluación (ROC-AUC) | Tabla comparativa, curvas ROC, matriz de confusión, Precision, Recall y F1 |
| Ranking de leads | Pestaña **Ranking**: leads ordenados por score, descarga en CSV |
| Sistema de priorización | Pestaña **Priorización**: niveles Caliente, Tibio y Frío, calificación de leads nuevos y por lote |
| Dashboard comercial | Pestaña **Dashboard**: KPIs, conversión por fuente y actividad, distribución de scores |

## 3. Metodología (CRISP-DM)

1. **Entendimiento del negocio:** objetivo SMART y métrica de éxito (ROC-AUC).
2. **Entendimiento de los datos (EDA):** balance de clases, nulos reales (`"Select"` = no respondió), outliers con IQR y percentiles, tasa de conversión por variable.
3. **Preparación de los datos:**

   | Procedimiento | Dimensión de calidad | Técnica |
   |---|---|---|
   | Duplicados | Unicidad | Verificado: 0 duplicados |
   | `"Select"` → nulo | Completitud | `replace` |
   | Eliminar columnas con **fuga de datos** | Validez | `Tags`, `Lead Quality`, `Lead Profile`, `Asymmetrique*` |
   | Eliminar columnas sin valor analítico | Reducción dimensional | Más del 40 % faltante o varianza casi nula |
   | Normalización de texto | Consistencia | `"google"` → `"Google"` |
   | Conversión de tipos | Integridad | Yes/No → 1/0 |
   | Nulos categóricos | Completitud | Categoría `"Desconocido"` |
   | Nulos numéricos | Completitud | Imputación por mediana |
   | Outliers | Estabilidad estadística | *Capping* al percentil 99 |
   | Categorías raras | Consistencia | Menos del 1 % → `"Otros"` |
   | Codificación | Preparación para modelado | One-hot encoding |
   | Escalado | Preparación para modelado | `StandardScaler` |

4. **Feature engineering:** `tiempo_por_visita` = tiempo en la web / (visitas + 1) y `visito_web` (1 si pasó tiempo en la web).
5. **Modelado:** división 80/20 estratificada; `Pipeline` de scikit-learn con preprocesamiento + modelo; ajuste de hiperparámetros con `RandomizedSearchCV` (validación cruzada estratificada de 5 pliegues, métrica ROC-AUC).
6. **Evaluación:** métricas en el 20 % de prueba, curvas ROC, matriz de confusión, control de sobreajuste e importancia de variables.
7. **Despliegue:** app de Streamlit publicada en Streamlit Community Cloud (gratis).

> **¿Por qué se eliminaron `Tags` y `Lead Quality`?** Las llena el equipo de ventas **después** de contactar al lead. Usarlas inflaría el ROC-AUC, pero en la vida real un lead nuevo todavía no las tiene. Esto se llama **fuga de datos (data leakage)**.

## 4. Resultados

| Modelo | ROC-AUC (test) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost (final)** | **0.891** | 0.822 | 0.763 | 0.780 | 0.771 |
| Gradient Boosting | 0.890 | 0.824 | 0.773 | 0.767 | 0.770 |
| Random Forest | 0.885 | 0.818 | 0.768 | 0.754 | 0.761 |
| Regresión Logística (base) | 0.878 | 0.800 | 0.760 | 0.705 | 0.731 |

- **Meta SMART cumplida:** ROC-AUC 0.891 ≥ 0.85.
- **Sin sobreajuste relevante:** ROC-AUC 0.935 en entrenamiento vs 0.891 en prueba.
- **Mejores hiperparámetros de XGBoost:** `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `min_child_weight=3`, `colsample_bytree=1.0`.
- **Matriz de confusión (1 848 leads de prueba, umbral 0.5):** TN = 964, FP = 172, FN = 157, TP = 555.
- **Variables más importantes:** origen *Lead Add Form*, ocupación, última actividad *SMS Sent* y tiempo en la web.

### Sistema de priorización (validado en datos de prueba)

| Nivel | Regla | Acción comercial | Conversión real |
|---|---|---|---|
| 🔥 Caliente | Score ≥ 80 | Llamar hoy (contacto en menos de 24 h) | **87.1 %** |
| 🌤️ Tibio | Score 50 a 79 | Nutrir: email y llamada en la semana | **64.7 %** |
| ❄️ Frío | Score < 50 | Campaña automática de email | **14.0 %** |

Llamando solo al **30 % de los leads mejor rankeados** se captura cerca del **64 %** de los clientes que compran.

## 5. La aplicación

| Pestaña | Qué muestra |
|---|---|
| **Dashboard** | Temperatura del pipeline, KPIs, conversión por fuente, efectividad del ranking, leads para llamar hoy, distribución de scores y conversión por última actividad |
| **Ranking** | Top 5 de leads y ranking completo con score, prioridad y acción sugerida; descarga en CSV |
| **Priorización** | Los 3 niveles con su conversión real, formulario para calificar un lead nuevo y carga de CSV por lotes |
| **Modelo** | Comparación de modelos, curvas ROC, matriz de confusión, importancia de variables, hiperparámetros y columnas eliminadas |

El menú lateral (flecha **»**) permite filtrar por fuente, origen, ocupación y prioridad.

Los scores del ranking se calcularon con `cross_val_predict` (5 pliegues): cada lead fue puntuado por un modelo que **no lo vio al entrenar**.

## 6. Estructura del proyecto

```
lead-scoring-app/
├── app.py                       # App Streamlit (dashboard, ranking, priorización, modelo)
├── lead_utils.py                # Limpieza, feature engineering y regla de prioridad (compartido)
├── requirements.txt             # Dependencias para el despliegue
├── .streamlit/config.toml       # Tema de la app
├── data/
│   ├── Leads.csv                # Dataset original (Kaggle)
│   └── leads_scored.csv         # Leads con score, prioridad y ranking (generado)
├── models/
│   ├── modelo_lead_scoring.pkl  # Pipeline final: preprocesamiento + XGBoost
│   ├── metricas.json            # Métricas, curvas ROC, matriz e importancia
│   └── categorias.json          # Categorías válidas por variable
└── notebooks/
    └── lead_scoring.ipynb       # EDA, limpieza, modelos, evaluación y conclusiones
```

## 7. Tecnologías

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Plotly · Streamlit · Streamlit Community Cloud · GitHub

## 8. Ejecutar en local

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`.

Para reentrenar el modelo, ejecuta `notebooks/lead_scoring.ipynb` (en Jupyter, VS Code o Google Colab subiendo `Leads.csv` y `lead_utils.py`). Si el `.pkl` no se puede abrir en otro sistema operativo, la app reentrena automáticamente el modelo con los mismos hiperparámetros.

## 9. Conclusiones

1. Los modelos de *boosting* superaron a la Regresión Logística base; XGBoost obtuvo el mejor ROC-AUC (0.891).
2. El modelo es honesto: se eliminaron las variables con fuga de datos y solo se usa información disponible cuando llega el lead.
3. El sistema de priorización separa con claridad a los leads: los Calientes convierten 6 veces más que los Fríos.
4. La app permite a ventas concentrar su esfuerzo en los leads con mayor probabilidad de compra.

---

Universidad Cooperativa de Colombia · Inteligencia de Negocios y Minería de Datos · 2026
