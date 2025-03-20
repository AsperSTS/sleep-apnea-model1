import numpy as np
import pandas as pd

df = pd.read_csv('shhs-harmonized-filtered.csv')

unique_values = {}
max_unique = 10  # Establecer el número máximo de valores únicos

for col in df.columns:
    unique_vals = df[col].unique()
    if len(unique_vals) > max_unique:
        unique_values[col] = unique_vals[:max_unique]  # Tomar solo los primeros 10
    else:
        unique_values[col] = unique_vals

# Crear DataFrame a partir de valores únicos
unique_df = pd.DataFrame({col: pd.Series(data) for col, data in unique_values.items()})

# Mostrar el DataFrame de valores únicos
print(unique_df)
unique_df.to_csv('shhs-harmonized-filtered-unique.csv', index=False)