# 🎯 Lead Scoring: predicción de conversión de leads

Proyecto final de **Inteligencia de Negocios y Minería de Datos** · Universidad Cooperativa de Colombia
Autor: David Fernando Ramírez de la Parra

Modelo predictivo que clasifica los leads de *X Education* según su probabilidad de conversión, desplegado en una app de Streamlit con **ranking de leads**, **sistema de priorización** y **dashboard comercial**.

## Resultados
| Modelo | ROC-AUC (test) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** (mejor) | **0.891** | 0.822 | 0.763 | 0.780 | 0.771 |
| Gradient Boosting | 0.890 | 0.824 | 0.773 | 0.767 | 0.770 |
| Random Forest | 0.885 | 0.818 | 0.768 | 0.754 | 0.761 |
| Regresión Logística (base) | 0.878 | 0.800 | 0.760 | 0.705 | 0.731 |

Priorización en datos de prueba: 🔥 Calientes convierten **87 %**, 🌤️ Tibios **65 %**, ❄️ Fríos **14 %**.

## Estructura
```
lead-scoring-app/
├── app.py                     # App Streamlit (ranking, priorización, dashboard, modelo)
├── lead_utils.py              # Limpieza y feature engineering (compartido notebook/app)
├── requirements.txt
├── data/
│   ├── Leads.csv              # Dataset original (Kaggle)
│   └── leads_scored.csv       # Leads con score (generado por el notebook)
├── models/
│   ├── modelo_lead_scoring.pkl
│   ├── metricas.json
│   └── categorias.json
└── notebooks/
    └── lead_scoring.ipynb     # EDA, limpieza, feature engineering, modelos, evaluación
```

## Metodología (CRISP-DM)
1. **Negocio:** priorizar leads para ventas. Objetivo SMART: ROC-AUC ≥ 0.85.
2. **Datos:** 9 240 leads × 37 variables; `Converted` es la variable objetivo (38.5 % positivos).
3. **Preparación:** 'Select' → nulo, eliminación de columnas con **fuga de datos** (`Tags`, `Lead Quality`, `Lead Profile`, `Asymmetrique*`), imputación, *capping* de outliers (p99), agrupación de categorías raras, one-hot encoding.
4. **Feature engineering:** `tiempo_por_visita`, `visito_web`.
5. **Modelado:** Regresión Logística, Random Forest, Gradient Boosting y XGBoost; ajuste con `RandomizedSearchCV` (CV estratificada 5 pliegues, métrica ROC-AUC).
6. **Evaluación:** ROC-AUC, curvas ROC, matriz de confusión, Precision, Recall, F1.
7. **Despliegue:** Streamlit Community Cloud (gratis).

## Ejecutar local
```
pip install -r requirements.txt
streamlit run app.py
```
