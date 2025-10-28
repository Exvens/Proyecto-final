
# Proyecto Final - Predicción de Precios de Casas 

Descripción
Este proyecto implementa un flujo completo de Machine Learning aplicado al dataset de la competencia de Kaggle House Prices: Advanced Regression Techniques.
El objetivo es predecir el precio de venta de casas a partir de características numéricas y categóricas, aplicando técnicas de EDA, preprocesamiento, modelado y despliegue interactivo con Streamlit.

Estructura del proyecto
Proyecto-final/
├─ data/
│  ├─ raw/            # Dataset original (train.csv, test.csv, data_description.txt)
│  └─ processed/      # Modelos entrenados y datos procesados
├─ notebooks/         # Espacio para análisis exploratorios en Jupyter
├─ src/
│  ├─ eda.py          # Análisis exploratorio con Plotly
│  ├─ preprocess.py   # Pipelines de preprocesamiento
│  ├─ models.py       # Entrenamiento y evaluación de modelos
│  ├─ utils.py        # Funciones auxiliares
│  └─ app_streamlit.py# Aplicación interactiva con Streamlit
├─ tests/
│  └─ test_pipeline.py# Pruebas unitarias
├─ requirements.txt   # Dependencias del proyecto
├─ README.md          # Documentación del proyecto
├─ .gitignore         # Archivos a ignorar en Git



Flujo de trabajo
1. Búsqueda y elección del dataset
- Dataset: House Prices: Advanced Regression Techniques (Kaggle).
- Justificación: dataset clásico de regresión, con variables numéricas y categóricas, ideal para aplicar técnicas de preprocesamiento y modelos de ML.
2. Análisis Exploratorio de Datos (EDA)
- Implementado en src/eda.py y en la app de Streamlit.
- Herramientas: Plotly para visualizaciones interactivas.
- Ejemplos:
- Histogramas de variables numéricas.
- Boxplots de SalePrice por variables categóricas.
- Matriz de correlación.
3. Preprocesamiento
- Implementado en src/preprocess.py.
- Técnicas aplicadas:
- Imputación de valores nulos (mediana para numéricas, moda para categóricas).
- Estandarización de variables numéricas.
- OneHotEncoding para variables categóricas.
- PCA para reducción de dimensionalidad (95% de varianza).
- Variables ordinales (OverallQual, OverallCond) tratadas como numéricas.
4. Modelos de Machine Learning
- Implementado en src/models.py.
- Modelos entrenados:
- Regresión Lineal (baseline).
- RandomForestRegressor (modelo más robusto).
- Métricas evaluadas:
- RMSE (Root Mean Squared Error).
- R² (Coeficiente de determinación).
- Se guarda automáticamente el mejor modelo en data/processed/.
5. Aplicación con Streamlit
- Implementado en src/app_streamlit.py.
- Funcionalidades:
- EDA interactivo: histogramas, boxplots, correlaciones.
- Predicción de precios: formulario para ingresar características de una casa y obtener el precio estimado usando el modelo entrenado.
6. Uso de GitHub
- Proyecto versionado con Git.
- Commits organizados por fases:
- "EDA con Plotly"
- "Preprocesamiento con Pipelines"
- "Entrenamiento de modelos"
- "App Streamlit con predicciones"

Instalación y ejecución
- Clonar el repositorio:
git clone <https://github.com/Exvens/Proyecto-final.git>
cd Proyecto-final
- Crear y activar entorno virtual:
python -m venv .venv
.venv\Scripts\Activate
- Instalar dependencias:
pip install -r requirements.txt
- Descargar dataset desde Kaggle:
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw
Expand-Archive data/raw/house-prices-advanced-regression-techniques.zip -DestinationPath data/raw
- Entrenar modelos:
python src/models.py
- Ejecutar la app de Streamlit:
streamlit run src/app_streamlit.py

Resultados
- Mejor modelo: Regresión Lineal (en este caso).
- Métricas obtenidas:
- RMSE ≈ 29,600 USD
- R² ≈ 0.885

Autores
- Proyecto desarrollado por Giovanny Obando Duque y Javier Eduardo Guerrero Buendia como entrega final de taller de Machine Learning.

