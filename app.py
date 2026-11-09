"""
Lead Scoring App - Streamlit
Ranking de leads · Sistema de priorización · Dashboard comercial
Ejecutar local:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import lead_utils as U

BASE = Path(__file__).parent
COLORES = {"🔥 Caliente": "#dc2626", "🌤️ Tibio": "#f59e0b", "❄️ Frío": "#3b82f6"}
ORDEN_PRIO = ["🔥 Caliente", "🌤️ Tibio", "❄️ Frío"]

st.set_page_config(page_title="Lead Scoring", page_icon="🎯", layout="wide")


# ---------- Carga (cacheada: se lee una sola vez) ----------
def entrenar_modelo():
    """Reentrena el mejor modelo (mismos hiperparámetros del notebook).
    Se usa si el .pkl no se puede abrir en otro sistema operativo o versión de librerías."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    met = json.load(open(BASE / "models" / "metricas.json", encoding="utf-8"))
    cats = json.load(open(BASE / "models" / "categorias.json", encoding="utf-8"))
    raw = pd.read_csv(BASE / "data" / "Leads.csv")
    d, _ = U.agrupar_raras(U.limpiar(raw), cats)
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), U.NUM_COLS),
        ("bin", "passthrough", U.BIN_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), U.CAT_COLS),
    ])
    params = met["mejores_parametros"].get("XGBoost", {})
    modelo = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **params)
    return Pipeline([("prep", prep), ("modelo", modelo)]).fit(d[U.FEATURES], d[U.TARGET])


@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    try:
        return joblib.load(BASE / "models" / "modelo_lead_scoring.pkl")
    except Exception:
        return entrenar_modelo()


@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / "data" / "leads_scored.csv")
    met = json.load(open(BASE / "models" / "metricas.json", encoding="utf-8"))
    cats = json.load(open(BASE / "models" / "categorias.json", encoding="utf-8"))
    return df, met, cats


modelo = cargar_modelo()
df, MET, CATS = cargar_datos()


def puntuar(raw: pd.DataFrame) -> pd.DataFrame:
    """Recibe leads con las columnas ORIGINALES del CSV y devuelve probabilidad y prioridad."""
    d, _ = U.agrupar_raras(U.limpiar(raw), CATS)
    prob = modelo.predict_proba(d[U.FEATURES])[:, 1]
    out = raw.copy()
    out["score"] = (prob * 100).round(1)
    out["prioridad"] = [U.prioridad(p) for p in prob]
    out["acción sugerida"] = out["prioridad"].map(U.ACCION)
    return out.sort_values("score", ascending=False)


# ---------- Barra lateral: filtros globales ----------
st.sidebar.title("🎯 Lead Scoring")
st.sidebar.caption(f"Modelo: **{MET['mejor_modelo']}** · ROC-AUC test: **{MET['auc_test']:.3f}**")
st.sidebar.divider()
st.sidebar.subheader("Filtros")
f_fuente = st.sidebar.multiselect("Fuente", sorted(df["fuente"].unique()))
f_origen = st.sidebar.multiselect("Origen", sorted(df["origen"].unique()))
f_ocup = st.sidebar.multiselect("Ocupación", sorted(df["ocupacion"].unique()))
f_prio = st.sidebar.multiselect("Prioridad", ORDEN_PRIO)

dff = df.copy()
if f_fuente:
    dff = dff[dff["fuente"].isin(f_fuente)]
if f_origen:
    dff = dff[dff["origen"].isin(f_origen)]
if f_ocup:
    dff = dff[dff["ocupacion"].isin(f_ocup)]
if f_prio:
    dff = dff[dff["prioridad"].isin(f_prio)]
st.sidebar.caption(f"{len(dff):,} de {len(df):,} leads")

st.title("🎯 Lead Scoring: predicción de conversión de leads")
st.caption("Clasifica cada lead según su probabilidad de convertirse en cliente, para que ventas priorice donde está el dinero.")

tab_dash, tab_rank, tab_prio, tab_mod = st.tabs(
    ["📊 Dashboard comercial", "🏆 Ranking de leads", "🎯 Priorización", "🧠 Modelo"])

# ======================= DASHBOARD =======================
with tab_dash:
    if dff.empty:
        st.warning("No hay leads con esos filtros.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Leads", f"{len(dff):,}")
        c2.metric("Conversión real", f"{dff['Converted'].mean()*100:.1f}%")
        c3.metric("Score promedio", f"{dff['score'].mean():.1f}")
        c4.metric("🔥 Calientes", f"{(dff['prioridad']=='🔥 Caliente').sum():,}")
        cal = dff[dff["prioridad"] == "🔥 Caliente"]
        c5.metric("Conversión de calientes", f"{cal['Converted'].mean()*100:.1f}%" if len(cal) else "-")

        col1, col2 = st.columns(2)
        with col1:
            cnt = dff["prioridad"].value_counts().reindex(ORDEN_PRIO).fillna(0).reset_index()
            cnt.columns = ["prioridad", "leads"]
            fig = px.pie(cnt, names="prioridad", values="leads", hole=0.55, color="prioridad",
                         color_discrete_map=COLORES, title="Leads por prioridad")
            st.plotly_chart(fig, width="stretch")
        with col2:
            conv = dff.groupby("prioridad")["Converted"].mean().reindex(ORDEN_PRIO).mul(100).round(1).reset_index()
            fig = px.bar(conv, x="prioridad", y="Converted", color="prioridad", color_discrete_map=COLORES,
                         text="Converted", title="Conversión real por prioridad (%)",
                         labels={"Converted": "% conversión", "prioridad": ""})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            t = dff.groupby("fuente").agg(leads=("Converted", "size"), conversion=("Converted", "mean"),
                                         score=("score", "mean")).reset_index()
            t["conversion"] = (t["conversion"] * 100).round(1)
            fig = px.bar(t.sort_values("conversion"), x="conversion", y="fuente", orientation="h",
                         color="score", color_continuous_scale="Blues", text="conversion",
                         hover_data=["leads"], title="Conversión por fuente (%)",
                         labels={"conversion": "% conversión", "fuente": "", "score": "Score prom."})
            st.plotly_chart(fig, width="stretch")
        with col4:
            t = dff.groupby("ultima_actividad").agg(leads=("Converted", "size"),
                                                   conversion=("Converted", "mean")).reset_index()
            t["conversion"] = (t["conversion"] * 100).round(1)
            fig = px.bar(t.sort_values("conversion"), x="conversion", y="ultima_actividad", orientation="h",
                         text="conversion", hover_data=["leads"], title="Conversión por última actividad (%)",
                         labels={"conversion": "% conversión", "ultima_actividad": ""},
                         color_discrete_sequence=["#16a34a"])
            st.plotly_chart(fig, width="stretch")

        col5, col6 = st.columns(2)
        with col5:
            fig = px.histogram(dff, x="score", color="prioridad", nbins=40, color_discrete_map=COLORES,
                               category_orders={"prioridad": ORDEN_PRIO},
                               title="Distribución de los scores", labels={"score": "Score (0-100)"})
            st.plotly_chart(fig, width="stretch")
        with col6:
            # Curva de captura: si ventas llama al top X% del ranking, ¿qué % de clientes captura?
            o = dff.sort_values("score", ascending=False)
            if o["Converted"].sum() > 0:
                x = np.arange(1, len(o) + 1) / len(o) * 100
                yv = o["Converted"].cumsum() / o["Converted"].sum() * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=yv, name="Con el modelo", line=dict(width=3, color="#2563eb")))
                fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Sin modelo (al azar)",
                                         line=dict(dash="dash", color="gray")))
                fig.update_layout(title="Curva de captura (ganancia)", xaxis_title="% de leads contactados (top del ranking)",
                                  yaxis_title="% de clientes capturados")
                st.plotly_chart(fig, width="stretch")

# ======================= RANKING =======================
with tab_rank:
    st.subheader("🏆 Ranking de leads por probabilidad de conversión")
    c1, c2 = st.columns([1, 1])
    solo_pend = c1.toggle("Solo leads pendientes (aún no convertidos)", value=True)
    top_n = c2.slider("Mostrar top", 10, 500, 50, step=10)
    r = dff[dff["Converted"] == 0] if solo_pend else dff
    r = r.sort_values("score", ascending=False).head(top_n)
    vista = r[["ranking", U.ID_COL, "score", "prioridad", "fuente", "origen", "ocupacion",
               "ultima_actividad", "tiempo_web", "visitas", "Converted"]].rename(columns={
        U.ID_COL: "Lead", "ultima_actividad": "última actividad", "tiempo_web": "seg. en la web",
        "Converted": "convirtió"})
    vista["acción sugerida"] = vista["prioridad"].map(U.ACCION)
    st.dataframe(vista, width="stretch", hide_index=True,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100,
                                                                         format="%.1f")})
    st.download_button("⬇️ Descargar ranking (CSV)", vista.to_csv(index=False).encode("utf-8"),
                       "ranking_leads.csv", "text/csv")
    st.caption("Los scores se calcularon con validación cruzada: cada lead fue puntuado por un modelo que no lo vio al entrenar.")

# ======================= PRIORIZACIÓN =======================
with tab_prio:
    st.subheader("🎯 Sistema de priorización")
    a, b, c = st.columns(3)
    for col, p in zip((a, b, c), ORDEN_PRIO):
        sub = dff[dff["prioridad"] == p]
        col.markdown(f"### {p}")
        col.markdown({"🔥 Caliente": "Score ≥ 80", "🌤️ Tibio": "Score 50 – 79", "❄️ Frío": "Score < 50"}[p])
        col.metric("Leads", f"{len(sub):,}")
        col.metric("Conversión real", f"{sub['Converted'].mean()*100:.1f}%" if len(sub) else "-")
        col.info(U.ACCION[p])

    st.divider()
    st.subheader("➕ Calificar un lead nuevo")
    with st.form("nuevo_lead"):
        f1, f2, f3 = st.columns(3)
        origen = f1.selectbox("Origen del lead", CATS["origen"])
        fuente = f1.selectbox("Fuente", CATS["fuente"])
        ocup = f1.selectbox("Ocupación", CATS["ocupacion"], index=CATS["ocupacion"].index("Unemployed"))
        visitas = f2.number_input("Visitas a la web", 0, 100, 3)
        tiempo = f2.number_input("Tiempo total en la web (segundos)", 0, 5000, 500)
        paginas = f2.number_input("Páginas vistas por visita", 0.0, 50.0, 2.0)
        act = f3.selectbox("Última actividad", CATS["ultima_actividad"],
                           index=CATS["ultima_actividad"].index("Email Opened"))
        esp = f3.selectbox("Especialización", CATS["especializacion"])
        ciudad = f3.selectbox("Ciudad", CATS["ciudad"])
        g1, g2 = st.columns(2)
        no_email = g1.checkbox("No desea recibir emails")
        copia = g2.checkbox("Pidió copia gratis de 'Mastering The Interview'")
        enviar = st.form_submit_button("Calcular score", type="primary")

    if enviar:
        raw = pd.DataFrame([{
            "Lead Origin": origen, "Lead Source": fuente, "Do Not Email": "Yes" if no_email else "No",
            "TotalVisits": visitas, "Total Time Spent on Website": tiempo, "Page Views Per Visit": paginas,
            "Last Activity": None if act == "Desconocido" else act,
            "Specialization": None if esp == "Desconocido" else esp,
            "What is your current occupation": None if ocup == "Desconocido" else ocup,
            "City": None if ciudad == "Desconocido" else ciudad,
            "A free copy of Mastering The Interview": "Yes" if copia else "No",
        }])
        res = puntuar(raw).iloc[0]
        r1, r2 = st.columns([1, 2])
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=res["score"], number={"suffix": "%"},
                                       title={"text": "Probabilidad de conversión"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORES[res["prioridad"]]},
                                              "steps": [{"range": [0, 50], "color": "#dbeafe"},
                                                        {"range": [50, 80], "color": "#fef3c7"},
                                                        {"range": [80, 100], "color": "#fee2e2"}]}))
        gauge.update_layout(height=260, margin=dict(t=50, b=10))
        r1.plotly_chart(gauge, width="stretch")
        r2.markdown(f"## {res['prioridad']}")
        r2.success(f"**Acción sugerida:** {res['acción sugerida']}")

    st.divider()
    st.subheader("📂 Calificar un lote de leads (CSV)")
    st.caption("Sube un CSV con las mismas columnas del dataset original de Kaggle (Leads.csv).")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            lote = puntuar(pd.read_csv(archivo))
            cols = [c for c in ["Lead Number", "Prospect ID"] if c in lote.columns] + ["score", "prioridad", "acción sugerida"]
            st.dataframe(lote[cols], width="stretch", hide_index=True)
            st.download_button("⬇️ Descargar resultados", lote.to_csv(index=False).encode("utf-8"),
                               "leads_calificados.csv", "text/csv")
        except KeyError as e:
            st.error(f"Al CSV le falta la columna {e}. Usa el formato de Leads.csv.")

# ======================= MODELO =======================
with tab_mod:
    st.subheader("🧠 Evaluación de los modelos")
    st.markdown(f"Mejor modelo: **{MET['mejor_modelo']}**. ROC-AUC train **{MET['auc_train']}** · "
                f"test **{MET['auc_test']}** (diferencia pequeña → sin overfitting relevante).")
    comp = pd.DataFrame(MET["comparacion"])
    st.dataframe(comp.style.format({c: "{:.4f}" for c in comp.columns if c != "Modelo"})
                 .highlight_max(subset=[c for c in comp.columns if c != "Modelo"], color="#bbf7d0"),
                 width="stretch", hide_index=True)

    m1, m2 = st.columns(2)
    with m1:
        fig = go.Figure()
        for nombre, cur in MET["curvas_roc"].items():
            auc = comp.set_index("Modelo").loc[nombre, "ROC-AUC"]
            fig.add_trace(go.Scatter(x=cur["fpr"], y=cur["tpr"], name=f"{nombre} ({auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar (0.5)", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="Curvas ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Recall",
                          legend=dict(x=0.45, y=0.05))
        st.plotly_chart(fig, width="stretch")
    with m2:
        cm = np.array(MET["matriz_confusion"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: No", "Pred: Sí"], y=["Real: No", "Real: Sí"],
                        title=f"Matriz de confusión ({MET['mejor_modelo']}, umbral 0.5)")
        st.plotly_chart(fig, width="stretch")

    imp = pd.Series(MET["importancia"]).sort_values()
    fig = px.bar(imp, orientation="h", title="Variables más importantes",
                 labels={"value": "Importancia", "index": ""}, color_discrete_sequence=["#2563eb"])
    fig.update_layout(showlegend=False, height=480)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Hiperparámetros encontrados (RandomizedSearchCV, CV 5 pliegues)"):
        st.json(MET["mejores_parametros"])
    with st.expander("Columnas eliminadas y motivo (incluye fuga de datos)"):
        st.table(pd.DataFrame.from_dict(U.DROP_REASONS, orient="index", columns=["Motivo"]))
