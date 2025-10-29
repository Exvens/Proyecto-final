import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# --- Cargar dataset automáticamente ---
df = pd.read_csv("data/raw/train.csv")

# --- Título de la app ---
st.title("Explorador interactivo y predicción de precios de casas")

st.write("Este dataset contiene información de la competencia de Kaggle *House Prices*.")

# --- Cargar modelo entrenado ---
MODEL_PATHS = [
    "data/processed/linear_regression.pkl",
    "data/processed/random_forest.pkl"
]
model = None
for path in MODEL_PATHS:
    try:
        model = joblib.load(path)
        st.success(f"Modelo cargado: {path}")
        break
    except Exception:
        pass

if model is None:
    st.error("No se encontró un modelo entrenado. Ejecuta 'python src/models.py' y vuelve a abrir la app.")

# ============================================================
# Sección 1: EDA interactivo
# ============================================================
st.header("Exploración interactiva (EDA)")

# --- Mostrar primeras filas ---
if st.checkbox("Ver primeras filas del dataset"):
    st.write(df.head())

# --- Selección de variable numérica para histograma ---
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
columna_num = st.selectbox("Selecciona una variable numérica para ver su distribución:", numeric_cols)

fig = px.histogram(df, x=columna_num, nbins=50, title=f"Distribución de {columna_num}")
st.plotly_chart(fig)

# --- Selección de variable categórica para boxplot ---
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
columna_cat = st.selectbox("Selecciona una variable categórica para comparar con SalePrice:", categorical_cols)

fig2 = px.box(df, x=columna_cat, y="SalePrice", title=f"Precio por {columna_cat}")
st.plotly_chart(fig2)

# --- Matriz de correlación ---
if st.checkbox("Mostrar matriz de correlación"):
    corr = df.corr(numeric_only=True)
    fig3 = px.imshow(corr, text_auto=True, title="Matriz de correlación")
    st.plotly_chart(fig3)

# ============================================================
# Sección 2: Predicción de precios
# ============================================================
st.header("Predicción de precio de una casa")

if model is not None:
    st.write("Completa los campos para generar una predicción. Usamos valores por defecto del dataset.")

    # Columnas que pediremos explícitamente
    cols_num_form = ["LotArea", "GrLivArea", "OverallQual", "OverallCond", "YearBuilt",
                     "FullBath", "BedroomAbvGr", "GarageCars"]
    cols_cat_form = ["Neighborhood", "HouseStyle", "KitchenQual", "Exterior1st"]

    input_data = {}

    # Layout en dos columnas
    col_left, col_right = st.columns(2)

    with col_left:
        for col in cols_num_form:
            if col in df.columns:
                default = float(df[col].median())
                input_data[col] = st.number_input(col, value=default, step=1.0)

    with col_right:
        for col in cols_cat_form:
            if col in df.columns:
                options = sorted(df[col].dropna().unique().tolist())
                default_idx = 0 if options else None
                input_data[col] = st.selectbox(col, options, index=default_idx if default_idx is not None else 0)

    # Construir DataFrame con todas las columnas que el modelo espera
    X_cols = df.drop(columns=["SalePrice"]).columns

    def default_for_col(c):
        if c in df.select_dtypes(include=["int64", "float64"]).columns:
            return float(df[c].median())
        elif c in df.select_dtypes(include=["object"]).columns:
            return df[c].mode()[0]
        else:
            return np.nan

    user_df = pd.DataFrame([{c: input_data.get(c, default_for_col(c)) for c in X_cols}])

    st.subheader("Datos de entrada al modelo")
    st.write(user_df)

    if st.button("Predecir precio"):
        try:
            pred = model.predict(user_df)[0]
            st.success(f"Precio estimado: {pred:,.0f} USD")
        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.info("Asegúrate de que el modelo fue entrenado con las mismas columnas.")