import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("mental_health_dataset_corregido.csv")

features = ["stress_level", "anxiety_score", "depression_score", "sleep_hours"]
df_clean = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

#Crear modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

#Agregar los clusters
df_clean["cluster"] = kmeans.labels_

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x="stress_level", y="anxiety_score", hue="cluster", palette="viridis")
plt.title("K-Means Clustering: Estrés vs Ansiedad")
plt.savefig("kmeans_clusters.png")
plt.show()
