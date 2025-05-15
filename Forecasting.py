import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("mental_health_dataset_corregido.csv")

df["survey_date"] = pd.to_datetime(df["survey_date"])

df = df.sort_values("survey_date")

#Variable dependiente
y = df["anxiety_score"]

df["date_ordinal"] = df["survey_date"].map(pd.Timestamp.toordinal)
X = df[["date_ordinal"]]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

#Crear fechas futuras
future_dates = pd.date_range(df["survey_date"].max(), periods=10, freq="D")
future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
future_preds = model.predict(future_ordinals)

plt.figure(figsize=(10, 5))
plt.plot(df["survey_date"], y, label="Datos reales")
plt.plot(df["survey_date"], y_pred, label="Tendencia", color="orange")
plt.plot(future_dates, future_preds, label="Predicción futura", color="red", linestyle="--")
plt.title("Forecast de Ansiedad usando Regresión Lineal")
plt.xlabel("Fecha")
plt.ylabel("Puntaje de Ansiedad")
plt.legend()
plt.grid(True)
plt.savefig("forecast_anxiety.png")
plt.show()
