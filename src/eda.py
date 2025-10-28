import pandas as pd
import plotly.express as px

df = pd.read_csv("data/raw/train.csv")
print(df.shape)
print(df.info())
print(df.describe())

# Histograma de precios
fig = px.histogram(df, x="SalePrice", nbins=50, title="Distribución de precios de casas")
fig.show()

# Boxplot por vecindario
fig2 = px.box(df, x="Neighborhood", y="SalePrice", title="Precio por vecindario")
fig2.show()

# Heatmap de correlaciones
corr = df.corr(numeric_only=True)
fig3 = px.imshow(corr, text_auto=True, title="Matriz de correlación")
fig3.show()