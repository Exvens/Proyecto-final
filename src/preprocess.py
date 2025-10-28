from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

numeric_features = df.select_dtypes(include=["int64","float64"]).columns.drop("SalePrice")
categorical_features = df.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

""" desde aqui """
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Cargar datos
df = pd.read_csv("data/raw/train.csv")

# Separar X e y
y = df["SalePrice"]
X = df.drop(columns=["SalePrice"])

# Definir columnas
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Ejemplo de variables ordinales (si existen): OverallQual, OverallCond
# Nota: Estas ya son numéricas; si tuvieran etiquetas, se transforman a orden (Label Encoding).
ordinal_cols = ["OverallQual", "OverallCond"]
# Las mantenemos en numeric_cols para escalado y PCA

# Transformadores
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))  # mantener el 95% de la varianza
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Funciones de utilidad para exportar
def get_features():
    return X, y, numeric_cols, categorical_cols, ordinal_cols

def get_preprocessor():
    return preprocessor

if __name__ == "__main__":
    # Prueba rápida del preprocesador
    Xt = preprocessor.fit_transform(X)
    print("Forma original X:", X.shape)
    print("Forma transformada Xt:", Xt.shape)