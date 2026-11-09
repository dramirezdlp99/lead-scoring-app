"""
Lead Scoring App - Streamlit
Ranking de leads, sistema de priorización y dashboard comercial.
Ejecutar local:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import lead_utils as U

BASE = Path(__file__).parent
st.set_page_config(page_title="LeadScore", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# Sistema de diseño (inspirado en un dashboard de Dribbble:
# fondo degradado cálido, tarjetas de vidrio, acento amarillo y carbón)
# ============================================================
INK = "#1F1F1F"
CARBON = "#2B2B2B"
MUTED = "#8A877F"
LINE = "#E6E2D8"
YEL = "#F5C935"
YEL_SOFT = "#FBE7A1"
GRIS = "#CFCAC0"

ORDEN_PRIO = ["🔥 Caliente", "🌤️ Tibio", "❄️ Frío"]
COLORES = {"🔥 Caliente": YEL, "🌤️ Tibio": CARBON, "❄️ Frío": GRIS}
NOMBRE = {"🔥 Caliente": "Calientes", "🌤️ Tibio": "Tibios", "❄️ Frío": "Fríos"}
SINGULAR = {"🔥 Caliente": "caliente", "🌤️ Tibio": "tibio", "❄️ Frío": "frío"}
RANGO = {"🔥 Caliente": "Score de 80 a 100", "🌤️ Tibio": "Score de 50 a 79", "❄️ Frío": "Score menor a 50"}
CORTO = {"Welingak Website": "Welingak", "Direct Traffic": "Directo", "Organic Search": "Orgánico",
         "Referral Sites": "Referidos", "Olark Chat": "Olark", "Reference": "Referencia", "Google": "Google",
         "Otros": "Otros"}
FUENTE = "Outfit, system-ui, sans-serif"

pio.templates["leads"] = go.layout.Template(layout=dict(
    font=dict(family=FUENTE, size=13, color=INK),
    title=dict(font=dict(family=FUENTE, size=18, color=INK, weight=400), x=0.02, xanchor="left", y=0.95),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[CARBON, YEL, GRIS, MUTED],
    xaxis=dict(gridcolor="rgba(0,0,0,0)", zerolinecolor=LINE, linecolor="rgba(0,0,0,0)",
               tickfont=dict(color=MUTED), automargin=True),
    yaxis=dict(gridcolor=LINE, gridwidth=1, griddash="dot", zerolinecolor=LINE, linecolor="rgba(0,0,0,0)",
               tickfont=dict(color=MUTED), automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    margin=dict(l=18, r=22, t=64, b=18),
    hoverlabel=dict(bgcolor=CARBON, bordercolor=CARBON, font=dict(family=FUENTE, color="white")),
))
pio.templates.default = "leads"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');

html, body, .stApp, .stMarkdown, button, input, textarea, select, label, p, li, span, div {
  font-family: 'Outfit', system-ui, sans-serif;
}
.stApp {
  background: linear-gradient(125deg, #E7E8EA 0%, #EFEEEA 38%, #F6F0DC 70%, #FBEFC4 100%);
  background-attachment: fixed; color: #1F1F1F;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1340px; }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; color: #1F1F1F; font-weight: 400; }

/* Barra lateral */
section[data-testid="stSidebar"] { background: rgba(255,255,255,.65); backdrop-filter: blur(14px); border-right: 0; }
section[data-testid="stSidebar"] h2 { font-weight: 400; font-size: 1.3rem; }

/* Navegación en píldoras (pestañas) */
div[data-testid="stTabs"] [role="tablist"] {
  background: rgba(255,255,255,.75); border-radius: 999px; padding: 6px; gap: 4px; width: fit-content;
  border: 1px solid rgba(255,255,255,.95); margin-bottom: 1.2rem; box-shadow: none;
}
div[data-testid="stTabs"] [role="tablist"]::before, div[data-testid="stTabs"] [role="tablist"]::after,
div[data-testid="stTabs"] [role="tab"]::before, div[data-testid="stTabs"] [role="tab"]::after { display: none !important; }
div[data-testid="stTabs"] [role="tablist"] > :not([role="tab"]) { display: none !important; }
div[data-testid="stTabs"] [role="tab"] {
  border-radius: 999px !important; padding: .5rem 1.25rem !important; margin: 0 !important; color: #4A4843;
  background: transparent; border: 0 !important; box-shadow: none !important; cursor: pointer;
}
div[data-testid="stTabs"] [role="tab"] p { font-size: .95rem; color: inherit !important; margin: 0; }
div[data-testid="stTabs"] [role="tab"]:hover { background: rgba(0,0,0,.05); }
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: #2B2B2B !important; color: #FFFFFF !important; }
div[data-testid="stTabs"] [role="tab"]:focus-visible { outline: 3px solid #F5C935; }

/* Tarjetas de vidrio para gráficas, tablas, formularios */
div[data-testid="stPlotlyChart"] {
  background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.95); border-radius: 26px;
  overflow: hidden; backdrop-filter: blur(10px);
}
div[data-testid="stDataFrame"] { border-radius: 18px; overflow: hidden; border: 1px solid rgba(255,255,255,.95); }
div[data-testid="stForm"] {
  background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.95); border-radius: 26px; padding: 1.5rem 1.6rem;
}
div[data-testid="stExpander"] details { background: rgba(255,255,255,.6); border: 1px solid rgba(255,255,255,.95); border-radius: 18px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"], .stNumberInput input { border-radius: 14px !important; }

/* Botones en píldora */
div[data-testid="stFormSubmitButton"] button, .stButton button {
  background: #2B2B2B; color: #fff; border: 0; border-radius: 999px; padding: .6rem 1.5rem; font-weight: 500;
}
div[data-testid="stFormSubmitButton"] button:hover { background: #F5C935; color: #1F1F1F; }
div[data-testid="stDownloadButton"] button { border-radius: 999px; background: rgba(255,255,255,.8); border: 1px solid #E6E2D8; }
button:focus-visible { outline: 3px solid #F5C935 !important; outline-offset: 2px; }

/* ---------- Encabezado ---------- */
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
.logo { border: 1.5px solid #2B2B2B; border-radius: 999px; padding: .35rem 1.1rem; font-size: 1.15rem; font-weight: 400; }
.pills { display: flex; gap: .5rem; flex-wrap: wrap; }
.pill { background: rgba(255,255,255,.75); border-radius: 999px; padding: .45rem 1rem; font-size: .88rem; color: #4A4843;
  border: 1px solid rgba(255,255,255,.95); }
.pill b { font-weight: 600; color: #1F1F1F; }

.hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.6rem; }
.hero h1 { font-size: clamp(2.2rem, 3.6vw, 3.1rem); font-weight: 300; letter-spacing: -0.02em; margin: 0 0 1rem; padding: 0; line-height: 1.05; }
.temp { display: flex; gap: 6px; align-items: flex-end; min-width: 320px; flex: 1; max-width: 640px; }
.temp .seg { display: flex; flex-direction: column; gap: 6px; min-width: 70px; }
.temp .lab { font-size: .82rem; color: #4A4843; white-space: nowrap; }
.temp .bar { height: 40px; border-radius: 999px; display: flex; align-items: center; padding: 0 14px; font-size: .82rem; font-weight: 500; }
.stats { display: flex; gap: 2.2rem; flex-wrap: wrap; }
.stat .n { font-size: clamp(2.8rem, 4.6vw, 4.1rem); font-weight: 300; letter-spacing: -0.04em; line-height: 1; }
.stat .k { font-size: .82rem; color: #6F6C64; margin-top: .3rem; display: flex; align-items: center; gap: .4rem; }
.stat .k i { width: 18px; height: 18px; border-radius: 50%; background: rgba(255,255,255,.85); display: inline-block; border: 1px solid #E6E2D8; }

/* ---------- Tarjetas HTML ---------- */
.card { background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.95); border-radius: 26px; padding: 1.2rem 1.35rem;
  height: 370px; backdrop-filter: blur(10px); display: flex; flex-direction: column; }
.card h4 { font-size: 1.12rem; font-weight: 400; margin: 0; padding: 0; color: inherit; }
.card .sub { font-size: .8rem; color: #8A877F; }
.card.dark { background: #2B2B2B; color: #fff; border-color: #2B2B2B; }
.card.dark .sub { color: #A8A59E; }
.card .head { display: flex; justify-content: space-between; align-items: flex-start; gap: .6rem; }
.card .big { font-size: 2.3rem; font-weight: 300; letter-spacing: -0.03em; line-height: 1; }

.lista { margin-top: 1rem; display: flex; flex-direction: column; gap: .75rem; }
.item { display: flex; align-items: center; gap: .8rem; }
.item .ic { width: 38px; height: 38px; border-radius: 50%; border: 1px solid #4A4A4A; display: flex; align-items: center;
  justify-content: center; font-size: .85rem; color: #fff; flex-shrink: 0; }
.item .ic.on { background: #fff; color: #2B2B2B; border-color: #fff; }
.item .tx { flex: 1; line-height: 1.2; }
.item .tx b { font-weight: 500; font-size: .92rem; }
.item .tx span { display: block; font-size: .76rem; color: #A8A59E; }
.item .dot { width: 18px; height: 18px; border-radius: 50%; background: #F5C935; flex-shrink: 0; }

.niveles { display: flex; gap: 6px; margin-top: auto; }
.niveles .col { display: flex; flex-direction: column; gap: 6px; }
.niveles .pc { font-size: .8rem; color: #4A4843; }
.niveles .bl { height: 46px; border-radius: 14px; display: flex; align-items: flex-end; padding: 0 0 8px 12px; font-size: .8rem; font-weight: 500; }
.rayas { background: repeating-linear-gradient(-45deg, #D8D3C8 0 2px, rgba(255,255,255,.6) 2px 8px); color: #4A4843; }

.tabla-p { margin-top: 1.3rem; display: flex; flex-direction: column; }
.fila-p { display: grid; grid-template-columns: 1.4fr 1fr 1fr; align-items: center; padding: .55rem 0;
  border-bottom: 1px dashed #E6E2D8; font-size: .9rem; }
.fila-p span:not(.nm) { text-align: right; font-weight: 500; }
.fila-p.enc { font-size: .76rem; color: #8A877F; padding-top: 0; }
.fila-p.enc span { font-weight: 400 !important; }
.fila-p.enc span:first-child { text-align: left; }
.fila-p .nm { display: flex; align-items: center; gap: .55rem; }
.fila-p .sq { width: 14px; height: 14px; border-radius: 5px; display: inline-block; flex-shrink: 0; }
.barra-p { display: flex; gap: 4px; height: 40px; margin-top: auto; }
.barra-p .seg-b { border-radius: 12px; min-width: 10px; }

.ring { display: flex; align-items: center; justify-content: center; flex: 1; }

/* Ranking */
.top5 { display: grid; grid-template-columns: repeat(5, 1fr); gap: .8rem; margin-bottom: 1.2rem; }
.lead { background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.95); border-radius: 22px; padding: 1rem 1.1rem; }
.lead.dark { background: #2B2B2B; color: #fff; border-color: #2B2B2B; }
.lead .pos { font-size: .78rem; color: #8A877F; }
.lead .sc { font-size: 2.3rem; font-weight: 300; letter-spacing: -0.03em; line-height: 1.1; }
.lead .id { font-weight: 500; font-size: .92rem; }
.lead .meta { font-size: .78rem; color: #8A877F; margin-top: .25rem; line-height: 1.35; }
.lead.dark .meta, .lead.dark .pos { color: #A8A59E; }
.lead .mini { height: 6px; border-radius: 6px; background: rgba(0,0,0,.08); margin-top: .7rem; overflow: hidden; }
.lead.dark .mini { background: rgba(255,255,255,.15); }
.lead .mini div { height: 100%; border-radius: 6px; background: #F5C935; }

/* Priorización */
.nivel { border-radius: 26px; padding: 1.4rem 1.5rem; height: 100%; }
.nivel h4 { margin: 0; font-size: 1.4rem; font-weight: 400; color: inherit; }
.nivel .r { font-size: .86rem; opacity: .75; }
.nivel .fila { display: flex; gap: 2rem; margin: 1.2rem 0; }
.nivel .fila .v { font-size: 2.4rem; font-weight: 300; letter-spacing: -0.03em; line-height: 1; }
.nivel .fila .k { font-size: .8rem; opacity: .75; }
.nivel .acc { border-radius: 999px; padding: .55rem 1rem; font-size: .88rem; display: inline-block; }

.resultado { background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.95); border-radius: 26px; padding: 1.2rem 1.4rem;
  display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
.resultado .t { font-size: 1.6rem; font-weight: 300; }
.resultado .a { display: inline-block; margin-top: .6rem; background: #2B2B2B; color: #fff; border-radius: 999px; padding: .5rem 1rem; font-size: .9rem; }

.seccion { margin: 1.4rem 0 .6rem; }
.seccion h3 { margin: 0; font-size: 1.45rem; font-weight: 400; padding: 0; }
.seccion p { color: #6F6C64; margin: .15rem 0 0; font-size: .93rem; }

@media (max-width: 900px) {
  .top5 { grid-template-columns: repeat(2, 1fr); }
  .card { height: auto; min-height: 320px; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Componentes
# ============================================================
def seccion(titulo, texto=""):
    st.markdown(f'<div class="seccion"><h3>{titulo}</h3>{f"<p>{texto}</p>" if texto else ""}</div>',
                unsafe_allow_html=True)


def anillo(pct, grande, pequeno, color=YEL, tam=200, fondo="#EFECE4", texto=INK):
    """Anillo de progreso en SVG con marcas alrededor (estilo 'time tracker')."""
    r, c = 62, 2 * np.pi * 62
    arco = max(min(pct, 100), 0) / 100 * c
    return f"""
    <svg viewBox="0 0 180 180" width="{tam}" height="{tam}" role="img" aria-label="{grande} {pequeno}">
      <circle cx="90" cy="90" r="82" fill="none" stroke="{GRIS}" stroke-width="7" stroke-dasharray="1.3 4.1"/>
      <circle cx="90" cy="90" r="{r}" fill="none" stroke="{fondo}" stroke-width="15"/>
      <circle cx="90" cy="90" r="{r}" fill="none" stroke="{color}" stroke-width="15" stroke-linecap="round"
              stroke-dasharray="{arco:.1f} {c:.1f}" transform="rotate(-90 90 90)"/>
      <text x="90" y="92" text-anchor="middle" font-family="Outfit, sans-serif" font-size="32" font-weight="300" fill="{texto}">{grande}</text>
      <text x="90" y="114" text-anchor="middle" font-family="Outfit, sans-serif" font-size="11" fill="{MUTED}">{pequeno}</text>
    </svg>"""


def grafica(fig, alto):
    fig.update_layout(height=alto)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


# ============================================================
# Carga de modelo y datos (cacheada)
# ============================================================
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


# ============================================================
# Filtros (menú lateral)
# ============================================================
with st.sidebar:
    st.markdown("## Filtrar leads")
    f_fuente = st.multiselect("Fuente", sorted(df["fuente"].unique()), placeholder="Todas")
    f_origen = st.multiselect("Origen", sorted(df["origen"].unique()), placeholder="Todos")
    f_ocup = st.multiselect("Ocupación", sorted(df["ocupacion"].unique()), placeholder="Todas")
    f_prio = st.multiselect("Prioridad", ORDEN_PRIO, placeholder="Todas")

dff = df.copy()
if f_fuente:
    dff = dff[dff["fuente"].isin(f_fuente)]
if f_origen:
    dff = dff[dff["origen"].isin(f_origen)]
if f_ocup:
    dff = dff[dff["ocupacion"].isin(f_ocup)]
if f_prio:
    dff = dff[dff["prioridad"].isin(f_prio)]
st.sidebar.caption(f"Mostrando {len(dff):,} de {len(df):,} leads")

# ============================================================
# Encabezado
# ============================================================
filtros_activos = sum(bool(f) for f in (f_fuente, f_origen, f_ocup, f_prio))
st.markdown(f"""
<div class="topbar">
  <div class="logo">LeadScore</div>
  <div class="pills">
    <div class="pill">Modelo <b>{MET['mejor_modelo']}</b></div>
    <div class="pill">ROC-AUC <b>{MET['auc_test']:.3f}</b></div>
    <div class="pill">{"Filtros activos: <b>" + str(filtros_activos) + "</b>" if filtros_activos else "Filtros en el menú lateral"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

if dff.empty:
    st.warning("Ningún lead cumple esos filtros. Quita alguno en el menú lateral.")
    st.stop()

total = len(dff)
pend = dff[dff["Converted"] == 0]
cal_pend = pend[pend["prioridad"] == "🔥 Caliente"].sort_values("score", ascending=False)

temp = ""
for p in ORDEN_PRIO:
    pct = (dff["prioridad"] == p).mean() * 100
    estilo = {"🔥 Caliente": f"background:{YEL};color:{INK}",
              "🌤️ Tibio": f"background:{CARBON};color:#fff",
              "❄️ Frío": ""}[p]
    clase = "bar rayas" if p == "❄️ Frío" else "bar"
    temp += (f'<div class="seg" style="flex:{max(pct, 12):.1f}"><div class="lab">{NOMBRE[p]}</div>'
             f'<div class="{clase}" style="{estilo}">{pct:.0f}%</div></div>')

st.markdown(f"""
<div class="hero">
  <div style="flex:1;min-width:320px">
    <h1>Hola, equipo de ventas</h1>
    <div class="temp">{temp}</div>
  </div>
  <div class="stats">
    <div class="stat"><div class="n">{total:,}</div><div class="k"><i></i>Leads</div></div>
    <div class="stat"><div class="n">{dff['Converted'].mean()*100:.0f}%</div><div class="k"><i></i>Conversión</div></div>
    <div class="stat"><div class="n">{len(cal_pend):,}</div><div class="k"><i></i>Calientes por llamar</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_dash, tab_rank, tab_prio, tab_mod = st.tabs(["Dashboard", "Ranking", "Priorización", "Modelo"])

# ============================================================
# DASHBOARD
# ============================================================
with tab_dash:
    c1, c2, c3, c4 = st.columns([1.25, 1, 1, 1.1], gap="small")

    # 1) Piruletas: conversión por fuente
    with c1:
        t = dff.groupby("fuente")["Converted"].agg(["mean", "size"]).reset_index()
        t = t[t["size"] >= 20].sort_values("mean", ascending=False).head(7)
        t["pct"] = t["mean"] * 100
        t["nom"] = t["fuente"].map(CORTO).fillna(t["fuente"])
        colores = [YEL if i == 0 else CARBON for i in range(len(t))]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t["nom"], y=t["pct"], width=0.07, marker_color=colores, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t["nom"], y=t["pct"], mode="markers", marker=dict(size=11, color=colores),
                                 customdata=t["size"], hovertemplate="%{x}<br>%{y:.1f}% conversión<br>%{customdata} leads<extra></extra>"))
        top = t.iloc[0]
        fig.add_annotation(x=top["nom"], y=top["pct"], text=f"{top['pct']:.0f}%", showarrow=False, yshift=22,
                           bgcolor=YEL, borderpad=5, font=dict(size=12, color=INK))
        fig.update_layout(title="Conversión por fuente", showlegend=False,
                          yaxis=dict(range=[0, 118], ticksuffix="%", showgrid=True, dtick=25),
                          xaxis=dict(tickangle=-35, tickfont=dict(size=11)))
        grafica(fig, 370)

    # 2) Anillo: clientes capturados en el top 30 %
    with c2:
        o = dff.sort_values("score", ascending=False)
        cap = o["Converted"].iloc[:max(int(len(o) * 0.3), 1)].sum() / max(o["Converted"].sum(), 1) * 100
        st.markdown(f"""
        <div class="card">
          <div class="head"><div><h4>Efectividad del ranking</h4>
          <div class="sub">Clientes que están en el top 30 % de la lista</div></div></div>
          <div class="ring">{anillo(cap, f"{cap:.0f}%", "de los clientes")}</div>
          <div class="sub" style="text-align:center">Al azar, el top 30 % tendría solo el 30 %</div>
        </div>""", unsafe_allow_html=True)

    # 3) Niveles de prioridad (estilo 'onboarding')
    with c3:
        cal = dff[dff["prioridad"] == "🔥 Caliente"]
        conv_cal = cal["Converted"].mean() * 100 if len(cal) else 0
        filas, barra = "", ""
        for p in ORDEN_PRIO:
            sub = dff[dff["prioridad"] == p]
            pct = len(sub) / len(dff) * 100
            conv = sub["Converted"].mean() * 100 if len(sub) else 0
            muestra = {"🔥 Caliente": f"background:{YEL}", "🌤️ Tibio": f"background:{CARBON}", "❄️ Frío": ""}[p]
            clase = "sq rayas" if p == "❄️ Frío" else "sq"
            filas += (f'<div class="fila-p"><span class="nm"><i class="{clase}" style="{muestra}"></i>{NOMBRE[p]}</span>'
                      f'<span>{pct:.0f}%</span><span>{conv:.0f}%</span></div>')
            if pct > 0:
                seg = "seg-b rayas" if p == "❄️ Frío" else "seg-b"
                barra += f'<div class="{seg}" style="flex:{pct:.2f};{muestra}" title="{NOMBRE[p]}: {pct:.1f}%"></div>'
        st.markdown(f"""
        <div class="card">
          <div class="head"><div><h4>Priorización</h4><div class="sub">Conversión real de los calientes</div></div>
          <div class="big">{conv_cal:.0f}%</div></div>
          <div class="tabla-p">
            <div class="fila-p enc"><span>Nivel</span><span>Leads</span><span>Compran</span></div>
            {filas}
          </div>
          <div class="barra-p">{barra}</div>
        </div>""", unsafe_allow_html=True)

    # 4) Tarjeta oscura: a quién llamar hoy
    with c4:
        items = ""
        for i, (_, x) in enumerate(cal_pend.head(5).iterrows(), start=1):
            items += (f'<div class="item"><div class="ic{" on" if i == 1 else ""}">{i}</div>'
                      f'<div class="tx"><b>Lead {int(x[U.ID_COL])}</b>'
                      f'<span>{CORTO.get(x["fuente"], x["fuente"])}, score {x["score"]:.1f}</span></div>'
                      f'<div class="dot"></div></div>')
        if not items:
            items = '<div class="sub" style="margin-top:1rem">No hay leads calientes pendientes con estos filtros.</div>'
        st.markdown(f"""
        <div class="card dark">
          <div class="head"><h4>Llamar hoy</h4><div class="big">{min(5, len(cal_pend))}/{len(cal_pend)}</div></div>
          <div class="sub">Calientes que aún no compran</div>
          <div class="lista">{items}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    d1, d2 = st.columns([1.15, 1], gap="small")
    with d1:
        fig = px.histogram(dff, x="score", color="prioridad", nbins=40, color_discrete_map=COLORES,
                           category_orders={"prioridad": ORDEN_PRIO}, labels={"score": "Score", "prioridad": ""})
        fig.for_each_trace(lambda tr: tr.update(name=NOMBRE[tr.name]))
        fig.update_traces(marker_line_width=0)
        fig.update_layout(title="Cómo se reparten los scores", bargap=0.18, yaxis_title=None,
                          legend=dict(orientation="h", y=1.08, x=1, xanchor="right", yanchor="bottom"))
        grafica(fig, 340)
    with d2:
        t = dff.groupby("ultima_actividad")["Converted"].agg(["mean", "size"]).reset_index()
        t = t[t["size"] >= 20].sort_values("mean").tail(8)
        t["pct"] = t["mean"] * 100
        colores = [YEL if i == len(t) - 1 else CARBON for i in range(len(t))]
        fig = go.Figure(go.Bar(x=t["pct"], y=t["ultima_actividad"], orientation="h", marker_color=colores,
                               marker_cornerradius=8, width=0.55, text=t["pct"].map("{:.0f}%".format),
                               textposition="outside", cliponaxis=False, customdata=t["size"],
                               hovertemplate="%{y}<br>%{x:.1f}% conversión<br>%{customdata} leads<extra></extra>"))
        fig.update_layout(title="Conversión según la última actividad",
                          xaxis=dict(range=[0, 112], ticksuffix="%", showgrid=False),
                          yaxis=dict(showgrid=False))
        grafica(fig, 340)

# ============================================================
# RANKING
# ============================================================
with tab_rank:
    a, b = st.columns([1, 1])
    solo_pend = a.toggle("Solo leads pendientes (aún no compran)", value=True)
    top_n = b.slider("Cantidad de leads en la tabla", 10, 500, 50, step=10)
    r = dff[dff["Converted"] == 0] if solo_pend else dff
    r = r.sort_values("score", ascending=False)

    seccion("Los 5 primeros de la lista", "Son los que ventas debería contactar hoy.")
    tarjetas = ""
    for i, (_, x) in enumerate(r.head(5).iterrows(), start=1):
        tarjetas += (f'<div class="lead{" dark" if i == 1 else ""}"><div class="pos">Puesto {i}</div>'
                     f'<div class="sc">{x["score"]:.1f}</div>'
                     f'<div class="id">Lead {int(x[U.ID_COL])}</div>'
                     f'<div class="meta">{x["fuente"]}<br>{x["ocupacion"]}</div>'
                     f'<div class="mini"><div style="width:{x["score"]}%"></div></div></div>')
    st.markdown(f'<div class="top5">{tarjetas}</div>', unsafe_allow_html=True)

    seccion("Ranking completo")
    vista = r.head(top_n)[["ranking", U.ID_COL, "score", "prioridad", "fuente", "origen", "ocupacion",
                           "ultima_actividad", "tiempo_web", "visitas", "Converted"]].rename(columns={
        "ranking": "Puesto", U.ID_COL: "Lead", "score": "Score", "prioridad": "Prioridad", "fuente": "Fuente",
        "origen": "Origen", "ocupacion": "Ocupación", "ultima_actividad": "Última actividad",
        "tiempo_web": "Segundos en la web", "visitas": "Visitas", "Converted": "Compró"})
    vista["Acción sugerida"] = vista["Prioridad"].map(U.ACCION)
    vista["Compró"] = vista["Compró"].map({1: "Sí", 0: "No"})
    vista["Visitas"] = vista["Visitas"].fillna(0).astype(int)
    st.dataframe(vista, width="stretch", hide_index=True, height=460,
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                                         format="%.1f", color="#F5C935")})
    st.download_button("Descargar ranking en CSV", vista.to_csv(index=False).encode("utf-8"),
                       "ranking_leads.csv", "text/csv")
    st.caption("Cada lead fue puntuado por un modelo que no lo vio al entrenar (validación cruzada de 5 pliegues).")

# ============================================================
# PRIORIZACIÓN
# ============================================================
with tab_prio:
    seccion("Los tres niveles de prioridad", "El score de cada lead define qué hace ventas con él.")
    cols = st.columns(3, gap="small")
    estilos = {
        "🔥 Caliente": (f"background:{YEL};color:{INK}", f"background:{CARBON};color:#fff"),
        "🌤️ Tibio": (f"background:{CARBON};color:#fff", f"background:{YEL};color:{INK}"),
        "❄️ Frío": ("background:rgba(255,255,255,.72);border:1px solid rgba(255,255,255,.95);color:#1F1F1F",
                   f"background:{CARBON};color:#fff"),
    }
    for col, p in zip(cols, ORDEN_PRIO):
        sub = dff[dff["prioridad"] == p]
        conv = f"{sub['Converted'].mean()*100:.0f}%" if len(sub) else "-"
        caja, accion = estilos[p]
        col.markdown(f"""
        <div class="nivel" style="{caja}">
          <h4>{NOMBRE[p]}</h4>
          <div class="r">{RANGO[p]}</div>
          <div class="fila">
            <div><div class="v">{len(sub):,}</div><div class="k">leads</div></div>
            <div><div class="v">{conv}</div><div class="k">conversión real</div></div>
          </div>
          <div class="acc" style="{accion}">{U.ACCION[p]}</div>
        </div>""", unsafe_allow_html=True)

    seccion("Calificar un lead nuevo", "Llena los datos del lead y el modelo calcula su score al instante.")
    with st.form("nuevo_lead"):
        f1, f2, f3 = st.columns(3, gap="medium")
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
        copia = g2.checkbox("Pidió la copia gratis de 'Mastering The Interview'")
        enviar = st.form_submit_button("Calcular score")

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
        color = YEL if res["prioridad"] == "🔥 Caliente" else (CARBON if res["prioridad"] == "🌤️ Tibio" else MUTED)
        st.markdown(f"""
        <div class="resultado">
          {anillo(res['score'], f"{res['score']:.1f}%", "probabilidad", color=color, tam=170)}
          <div>
            <div class="t">Lead {SINGULAR[res['prioridad']]}</div>
            <div class="a">{res['acción sugerida']}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    seccion("Calificar varios leads a la vez",
            "Sube un CSV con las mismas columnas del archivo original Leads.csv de Kaggle.")
    archivo = st.file_uploader("Archivo CSV", type="csv", label_visibility="collapsed")
    if archivo is not None:
        try:
            lote = puntuar(pd.read_csv(archivo))
            cols_l = [c for c in ["Lead Number", "Prospect ID"] if c in lote.columns] + ["score", "prioridad", "acción sugerida"]
            st.dataframe(lote[cols_l], width="stretch", hide_index=True)
            st.download_button("Descargar leads calificados", lote.to_csv(index=False).encode("utf-8"),
                               "leads_calificados.csv", "text/csv")
        except KeyError as e:
            st.error(f"Al CSV le falta la columna {e}. Usa las columnas de Leads.csv y vuelve a subirlo.")

# ============================================================
# MODELO
# ============================================================
with tab_mod:
    seccion("Comparación de modelos",
            f"Ganó {MET['mejor_modelo']}. ROC-AUC de {MET['auc_train']} en entrenamiento y {MET['auc_test']} en prueba: "
            "la diferencia es pequeña, así que no hay sobreajuste relevante.")
    comp = pd.DataFrame(MET["comparacion"])
    st.dataframe(comp.style.format({c: "{:.4f}" for c in comp.columns if c != "Modelo"})
                 .highlight_max(subset=[c for c in comp.columns if c != "Modelo"], color=YEL_SOFT),
                 width="stretch", hide_index=True)
    st.write("")

    m1, m2 = st.columns(2, gap="small")
    with m1:
        fig = go.Figure()
        estilos_roc = {MET["mejor_modelo"]: dict(color=YEL, width=3.5)}
        otros = iter([dict(color=CARBON, width=1.8), dict(color=MUTED, width=1.6), dict(color=GRIS, width=1.6)])
        for nombre, cur in MET["curvas_roc"].items():
            auc = comp.set_index("Modelo").loc[nombre, "ROC-AUC"]
            fig.add_trace(go.Scatter(x=cur["fpr"], y=cur["tpr"], name=f"{nombre} ({auc:.3f})",
                                     line=estilos_roc.get(nombre) or next(otros)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar (0.500)", line=dict(dash="dot", color=GRIS, width=1.2)))
        fig.update_layout(title="Curvas ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Recall",
                          legend=dict(x=0.4, y=0.05))
        grafica(fig, 420)
    with m2:
        cm = np.array(MET["matriz_confusion"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#FFF8E1"], [1, YEL]],
                        x=["Predijo: no compra", "Predijo: compra"], y=["Real: no compró", "Real: compró"])
        fig.update_traces(textfont=dict(size=26, family=FUENTE), xgap=6, ygap=6)
        fig.update_layout(title=f"Matriz de confusión ({MET['mejor_modelo']}, umbral 0.5)", coloraxis_showscale=False)
        grafica(fig, 420)

    imp = pd.Series(MET["importancia"]).sort_values()
    colores = [YEL if i == len(imp) - 1 else CARBON for i in range(len(imp))]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=colores,
                           marker_cornerradius=6, width=0.6))
    fig.update_layout(title="Variables que más pesan en la predicción", xaxis_title="Importancia",
                      yaxis=dict(showgrid=False))
    grafica(fig, 500)
    with st.expander("Hiperparámetros encontrados (RandomizedSearchCV, validación cruzada de 5 pliegues)"):
        st.json(MET["mejores_parametros"])
    with st.expander("Columnas eliminadas y motivo (incluye fuga de datos)"):
        st.table(pd.DataFrame.from_dict(U.DROP_REASONS, orient="index", columns=["Motivo"]))
