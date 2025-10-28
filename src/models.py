from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from preprocess import preprocessor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

from preprocess import get_features, get_preprocessor

df = pd.read_csv("data/raw/train.csv")
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: Regresión Lineal
lr = Pipeline(steps=[("preprocessor", preprocessor),
                     ("model", LinearRegression())])
lr.fit(X_train, y_train)
print("MSE Linear:", mean_squared_error(y_test, lr.predict(X_test)))

# Modelo 2: Random Forest
rf = Pipeline(steps=[("preprocessor", preprocessor),
                     ("model", RandomForestRegressor(n_estimators=100, random_state=42))])
rf.fit(X_train, y_train)
print("MSE RF:", mean_squared_error(y_test, rf.predict(X_test)))

""" desde aqui """

# Cargar datos y preprocesador
X, y, numeric_cols, categorical_cols, ordinal_cols = get_features()
preprocessor = get_preprocessor()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo 1: Regresión Lineal
lr_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Modelo 2: Random Forest
rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# Entrenar
lr_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

# Evaluar
def evaluate(name, model):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, preds)
    print(f"{name} -> RMSE: {rmse:.2f} | R²: {r2:.3f}")
    return rmse, r2

lr_rmse, lr_r2 = evaluate("LinearRegression", lr_pipe)
rf_rmse, rf_r2 = evaluate("RandomForest", rf_pipe)

# Seleccionar y guardar el mejor
best_model = rf_pipe if rf_rmse < lr_rmse else lr_pipe
best_name = "random_forest.pkl" if best_model is rf_pipe else "linear_regression.pkl"

joblib.dump(best_model, f"data/processed/{best_name}")
print(f"Modelo guardado: data/processed/{best_name}")