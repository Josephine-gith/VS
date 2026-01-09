import pandas as pd

# Chemin vers le fichier Excel
file_path = "dambreak_experimental_data.xlsx"

# Colonnes finales souhaitées
final_columns = [
    "material",
    "Bn",
    "1 - h_inf/h0",
    "r [kg/m³]",
    "k [Pa s]",
    "h0 [m]",
    "r0 [m]",
]

# Lire toutes les feuilles
sheets = pd.read_excel(file_path, sheet_name=None, header=1)

# Liste pour stocker les DataFrames de chaque feuille
dfs = []

for sheet_name, df in sheets.items():
    # Copie pour éviter les modifications en place
    df = df.copy()

    # Si le matériau correspond au nom de la feuille

    # Garder uniquement les colonnes nécessaires
    df = df[final_columns]

    dfs.append(df)

# Concaténer toutes les feuilles en un seul DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Affichage
# print(final_df.head())
