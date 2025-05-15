import pandas as pd

# Cargar tu dataset limpio
df = pd.read_csv("mental_health_dataset_corregido.csv")

print(df.describe())
print(df.describe(include='all'))


# Agrupar por género
print(df.groupby('gender').mean(numeric_only=True))

# Agrupar por estado laboral
print(df.groupby('employment_status').mean(numeric_only=True))
