# Résultat de simulation : étalement en dam break

import pandas as pd

resultats = []

for i in range(401):
    fichier = f"ResultsParaview/Results/results_{i}.csv"
    
    try:
        df = pd.read_csv(fichier)
        resultats.append([df.iloc[0,0],df.iloc[0,1]])
    except Exception as e:
        print(f"Erreur avec {fichier} : {e}")

# Conversion en array numpy si besoin
import numpy as np
resultats = np.array(resultats)

pd.DataFrame(resultats, columns=["Time", "R_Max"]).to_csv("ResultsParaview/ResultsSim.csv", index=False)
