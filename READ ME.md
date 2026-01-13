# Viscoplastic Squeeze

## Modèle de Kelvin Voigt

### Implémentation du modèle

modeleKV.py : 
* Résoud le système avec une méthode d'Euler explicite. 
* Trace le rayon, la vitesse et la déformation en fonction du temps.

modeleKV_solve.py :
* Résoud le système avec solve_ivp de scipy.integrate.
* Trace le rayon, la vitesse et la déformation en fonction du temps.

### Visualisation des différents régimes et lois d'échelle
En dam break, c'est-à-dire sans plaque supérieure.

modeleKV_VPE_Contraintes.py :
* Calcule la contribution de chaque contrainte pour le régime visqueux, le régime plastique, le régime élastique et un régime mixte. On se met dans un régime en nulifiant les autres contribution (par exemple, pour le régime visqueux, on met $\tau_0\approx 0$ et $G\approx 0$).
* Trace la contribution de ces contraintes (valeur de la contrainte, normalisée par la contrainte principale - le poids) dans chaque cas.
* Permet une première vérification du modèle : cohérence entre les contraintes principales et le régime de dissipation.

modeleKV_VPE_LoiEchelle.py :
* Calcule le rayon final dans chacun des trois régimes, pour différentes valeurs de hauteur initiale de colonne.
* Trace pour chaque régime, le rayon final (ou maximal pour le régime élastique, puisque le rayon ne converge pas physiquement) normalisé, en fonction de $(Ga\frac{h_0}{r_0})^{\frac{1}{2m+3}}$ pour le régime visqueux, de $(\frac{1}{Bn}\frac{h_0}{r_0})^{\frac{1}{3}}$ pour le régime plastique et de $(\frac{1}{El}\frac{h_0}{r_0})^{\frac{1}{6}}$ pour le régime élastique. 
* L'obtention de droites permet de valider la cohérence du modèle avec la littérature, et l'expression de la loi d'échelle élastique.

modeleKV_VP_Regimes.py :
* Utilise la méthode solve_ivp de scipy.integrate pour calculer le rayon final pour différentes valeurs de $\tau_0$.
* Trace $\frac{r_\infty}{r_0}/(Ga\frac{h_0}{r_0})^{\frac{1}{2m+3}}$ en fonction de $(\frac{1}{Bn}\frac{h_0}{r_0})^{\frac{1}{3}}/(Ga\frac{h_0}{r_0})^{\frac{1}{2m+3}}$ pour visualiser les deux régimes et vérifier les lois d'échelle.

### Comparaison avec des données expérimentales, issues de la littérature

ExpData_Df.py :
* Récupère les données expérimentales stockées dans le tableau Excel dambreak_experimental_data.xlsx
* Les stocke dans un DataFrame

modeleKV_ExpData.py :
* Calcule les données finales pour chaque expérience.
* Trace la hauteur normalisée calculée par le modèle et celle donnée par l'expérience, pour chaque expérience (et matériau), en fonction du rapport de vitesse caractéristique $u_c$ par la vitesse de transition de régime $u_{crossover}$.
* Trace l'erreur entre la hauteur normalisée expérimentale et celle donnée par le modèle, en fonction de $u_c/u_{crossover}$.

## Modèle de Maxwell

modeleM.py :
* Résoud le système avec une méthode Euler explicite.
* Trace le rayon, la vitesse et la déformation en fonction du temps.
* Diverge.

modeleM2.py :
* Version régularisée du modèle précédent, pour essayer de diminuer la divergence.
* Diverge quand même.