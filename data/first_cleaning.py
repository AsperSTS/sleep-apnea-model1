import pandas as pd
from typing import List

def remove_unused_cols(dataset_filename: str, dictionary_filename: str)-> pd.DataFrame:
    dataframe_harmonized = pd.read_csv(dataset_filename)
    dictonary = pd.read_csv(dictionary_filename)
    dictonary_codes = dictonary["CODE"].tolist()  # Convertir a lista para una búsqueda más eficiente

    # Filtrar las columnas del dataframe_harmonized
    columns_to_keep = [col for col in dataframe_harmonized.columns if col in dictonary_codes]

    # Mantener solo las columnas filtradas
    dataframe_filtered = dataframe_harmonized[columns_to_keep]

    # Opcional: Imprimir las columnas resultantes o guardar el dataframe filtrado
    print(dataframe_filtered.columns)  # Para verificar las columnas resultantes
    print(dictonary)
    print(f"Columnas filtradas: {len(dataframe_filtered.columns)}")
    # dataframe_filtered.to_csv("shhs-harmonized-filtered.csv", index=False) # Para guardar el dataframe filtrado
    
    return dataframe_filtered
def main() -> None:
    
    dataset_filtered = remove_unused_cols("shhs-harmonized-dataset-0.21.0.csv","variables_dict.csv" )
    dataset_filtered.to_csv("shhs-harmonized-filtered.csv", index=False)
    
if __name__ == "__main__":
    main()