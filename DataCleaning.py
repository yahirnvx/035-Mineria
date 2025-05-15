import pandas as pd
import random
from datetime import datetime, timedelta

# Carga el dataset
df = pd.read_csv("mental_health_dataset.csv")

# Verifica columnas y tipos
print(df.info())

# Agrega columna de fecha (si no la tiene)
start_date = datetime(2023, 1, 1)
df['survey_date'] = [start_date + timedelta(days=random.randint(0, 364)) for _ in range(len(df))]

# Asegúrate de que sea tipo datetime
df['survey_date'] = pd.to_datetime(df['survey_date'])

# Verifica que haya al menos:
# - 2 columnas numéricas
# - 1 alfanumérica (object)
# - 1 fecha (datetime64)
print(df.dtypes)

# Opcional: limpiar datos nulos
df = df.dropna()

# Guarda dataset limpio
df.to_csv("mental_health_dataset_corregido.csv", index=False)
