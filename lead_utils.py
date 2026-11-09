"""
lead_utils.py
Funciones compartidas entre el notebook (entrenamiento) y la app (Streamlit).
Así la limpieza y el feature engineering son EXACTAMENTE iguales al entrenar y al predecir.
"""
import numpy as np
import pandas as pd

TARGET = "Converted"
ID_COL = "Lead Number"

# Columnas eliminadas y por qué (para sustentar ante el profe)
DROP_REASONS = {
    "Prospect ID": "Identificador único, no aporta patrón",
    "Tags": "FUGA DE DATOS: la llena ventas después de contactar al lead",
    "Lead Quality": "FUGA DE DATOS: calificación manual del vendedor",
    "Lead Profile": "FUGA DE DATOS + 74% nulos/'Select'",
    "Asymmetrique Activity Index": "Puntaje externo posterior, 46% nulos",
    "Asymmetrique Profile Index": "Puntaje externo posterior, 46% nulos",
    "Asymmetrique Activity Score": "Puntaje externo posterior, 46% nulos",
    "Asymmetrique Profile Score": "Puntaje externo posterior, 46% nulos",
    "Last Notable Activity": "Redundante con 'Last Activity'",
    "How did you hear about X Education": "79% nulos/'Select'",
    "Country": "27% nulos y 96% de los datos es India (sin varianza)",
    "What matters most to you in choosing a course": "99.9% un solo valor",
    "Do Not Call": "Varianza casi nula (2 'Yes')",
    "Search": "Varianza casi nula",
    "Magazine": "Un solo valor",
    "Newspaper Article": "Varianza casi nula",
    "X Education Forums": "Varianza casi nula",
    "Newspaper": "Varianza casi nula",
    "Digital Advertisement": "Varianza casi nula",
    "Through Recommendations": "Varianza casi nula",
    "Receive More Updates About Our Courses": "Un solo valor",
    "Update me on Supply Chain Content": "Un solo valor",
    "Get updates on DM Content": "Un solo valor",
    "I agree to pay the amount through cheque": "Un solo valor",
}

# Nombres en español para el modelo y la app
RENAME = {
    "Lead Origin": "origen",
    "Lead Source": "fuente",
    "Do Not Email": "no_email",
    "TotalVisits": "visitas",
    "Total Time Spent on Website": "tiempo_web",
    "Page Views Per Visit": "paginas_por_visita",
    "Last Activity": "ultima_actividad",
    "Specialization": "especializacion",
    "What is your current occupation": "ocupacion",
    "City": "ciudad",
    "A free copy of Mastering The Interview": "copia_gratis",
}

NUM_COLS = ["visitas", "tiempo_web", "paginas_por_visita",
            "tiempo_por_visita", "visito_web"]
BIN_COLS = ["no_email", "copia_gratis"]
CAT_COLS = ["origen", "fuente", "ultima_actividad", "especializacion",
            "ocupacion", "ciudad"]
FEATURES = NUM_COLS + BIN_COLS + CAT_COLS

# Límites para recortar outliers (percentil 99 calculado en el EDA del notebook)
CAPS = {"visitas": 17, "paginas_por_visita": 9}

# Categorías con menos de este % se agrupan en "Otros"
RARE_MIN = 0.01


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza + feature engineering. Recibe el CSV original (o un lead nuevo)."""
    d = df.copy()

    # 1) 'Select' significa que el usuario no eligió nada -> es un nulo
    d = d.replace("Select", np.nan)

    # 2) Eliminar columnas con fuga de datos, sin varianza o con demasiados nulos
    d = d.drop(columns=[c for c in DROP_REASONS if c in d.columns])

    # 3) Renombrar a español
    d = d.rename(columns=RENAME)

    # 4) Normalización de texto: 'google' y 'Google' son la misma fuente
    if "fuente" in d:
        d["fuente"] = d["fuente"].replace({"google": "Google",
                                           "Facebook": "Social Media"})

    # 5) Binarias Yes/No -> 1/0
    for c in BIN_COLS:
        if c in d:
            d[c] = d[c].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # 6) Nulos categóricos -> 'Desconocido' (el no responder también es información)
    for c in CAT_COLS:
        if c in d:
            d[c] = d[c].fillna("Desconocido").astype(str)

    # 7) Nulos numéricos -> mediana (los imputa el pipeline); aquí outliers recortados
    for c, cap in CAPS.items():
        if c in d:
            d[c] = d[c].clip(upper=cap)

    # 8) FEATURE ENGINEERING: variables nuevas
    visitas = d["visitas"].fillna(0)
    d["tiempo_por_visita"] = d["tiempo_web"] / (visitas + 1)
    d["visito_web"] = (d["tiempo_web"] > 0).astype(int)

    return d


def agrupar_raras(d: pd.DataFrame, categorias_validas: dict = None):
    """Agrupa categorías poco frecuentes en 'Otros'.
    Al entrenar devuelve el diccionario de categorías válidas; al predecir lo recibe."""
    d = d.copy()
    if categorias_validas is None:
        categorias_validas = {}
        for c in CAT_COLS:
            freq = d[c].value_counts(normalize=True)
            categorias_validas[c] = sorted(freq[freq >= RARE_MIN].index.tolist())
    for c in CAT_COLS:
        d[c] = d[c].where(d[c].isin(categorias_validas[c]), "Otros")
    return d, categorias_validas


def prioridad(prob: float) -> str:
    """Sistema de priorización comercial."""
    if prob >= 0.80:
        return "🔥 Caliente"
    if prob >= 0.50:
        return "🌤️ Tibio"
    return "❄️ Frío"


ACCION = {
    "🔥 Caliente": "Llamar hoy (contacto en < 24 h)",
    "🌤️ Tibio": "Nutrir: email + llamada en la semana",
    "❄️ Frío": "Campaña automática de email",
}
