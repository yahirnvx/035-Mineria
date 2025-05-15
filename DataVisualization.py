import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
df = pd.read_csv("mental_health_dataset_corregido.csv")

# Estilo general
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

#Distribución de estrés
plt.figure()
sns.histplot(df["stress_level"], bins=10, kde=True)
plt.title("Distribución del nivel de estrés")
plt.xlabel("Nivel de Estrés")
plt.ylabel("Frecuencia")
plt.savefig("hist_stress.png")

#Diagrama de pastel - Género
plt.figure()
df["gender"].value_counts().plot.pie(autopct="%1.1f%%", colors=["skyblue", "lightgreen", "salmon"])
plt.title("Distribución por género")
plt.ylabel("")
plt.savefig("pie_gender.png")

#Boxplot - Sueño vs Género
plt.figure()
sns.boxplot(data=df, x="gender", y="sleep_hours")
plt.title("Horas de sueño por género")
plt.savefig("boxplot_sleep.png")

#Scatter plot - Depresión vs Ansiedad
plt.figure()
sns.scatterplot(data=df, x="depression_score", y="anxiety_score", hue="gender")
plt.title("Depresión vs Ansiedad")
plt.xlabel("Score de Depresión")
plt.ylabel("Score de Ansiedad")
plt.savefig("scatter_depr_anx.png")

#Múltiples histogramas en bucle
numeric_cols = ["stress_level", "sleep_hours", "depression_score", "anxiety_score"]
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Histograma de {col}")
    plt.savefig(f"hist_{col}.png")

plt.show()
