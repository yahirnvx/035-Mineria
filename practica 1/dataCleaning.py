import pandas as pd
import random
from datetime import datetime, timedelta

df = pd.read_csv("mental_health_dataset.csv")
print(df.shape)
print(df.head())

# Ver información general
print(df.info())

# Ver nombres de columnas
print(df.columns)

# Ver estadísticas básicas
print(df.describe())

# Ver cuántos valores nulos hay por columna
print(df.isnull().sum())

# Opción A: Eliminar filas con datos nulos
df = df.dropna()

# Definir fecha de inicio
start_date = datetime(2023, 1, 1)

df['survey_date'] = [start_date + timedelta(days=random.randint(0, 364)) for _ in range(len(df))]

print(df[['survey_date']].head())
print(df.dtypes)

df['survey_date'] = pd.to_datetime(df['survey_date'])
df = df.drop_duplicates()

print(df['gender'].unique())

df['gender'] = df['gender'].str.strip().str.lower()

df.to_csv("mental_health_dataset_limpio.csv", index=False)
