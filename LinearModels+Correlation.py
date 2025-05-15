import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("mental_health_dataset_corregido.csv")

x = df[["depression_score"]]
y = df["anxiety_score"]

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

#R² score
r2 = r2_score(y, y_pred)
print(f"R² score: {r2:.4f}")

correlation = df["depression_score"].corr(df["anxiety_score"])
print(f"Correlación Pearson: {correlation:.4f}")

#Gráfico de regresión
plt.figure(figsize=(8, 5))
sns.regplot(x="depression_score", y="anxiety_score", data=df, line_kws={"color": "red"})
plt.title("Modelo Lineal: Depresión vs Ansiedad")
plt.xlabel("Score de Depresión")
plt.ylabel("Score de Ansiedad")
plt.savefig("modelo_lineal.png")
plt.show()
