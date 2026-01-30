# Viscoplastic Squeeze

## Modèle de Kelvin Voigt

### Implémentation du modèle

modeleKV.py : 
* Résoud le système avec une méthode d'Euler explicite. 
* Trace le rayon, la vitesse et la déformation en fonction du temps.

modeleKV_solve.py :
* Résoud le système avec solve_ivp de scipy.integrate.
* Trace le rayon, la vitesse (ou la hauteur) en fonction du temps, pour plusieurs valeurs de M.

modeleKV_solve_GlissAdh.py :
* Comme modeleKV_solve.py, mais pour le cas de glissement aussi.
* Trace le rayon et la hauteur en glissement et en non-glissement en fonction du temps, pour plusieurs valeurs de M.


### Visualisation des différents régimes et lois d'échelle

ContributionContraintes_VPE.py :
* En dam break.
* Calcule la contribution de chaque contrainte pour le régime visqueux, le régime plastique, le régime élastique et un régime mixte. On se met dans un régime en nulifiant les autres contribution (par exemple, pour le régime visqueux, on met $\tau_0\approx 0$ et $G\approx 0$).
* Trace la contribution de ces contraintes (valeur de la contrainte, normalisée par la contrainte principale - le poids) dans chaque cas.
* Permet une première vérification du modèle : cohérence entre les contraintes principales et le régime de dissipation.

* ContributionContraintes_GlissAdh.py :
* Calcule et trace la contribution de chaque contrainte au cours d'un étalement, glissant ou non-glissant.

LoiEchelle.py :
* Calcule le rayon final dans chacun des trois régimes, pour différentes valeurs de hauteur initiale de colonne.
* Trace pour chaque régime, le rayon final (ou maximal pour le régime élastique, puisque le rayon ne converge pas physiquement) normalisé, en fonction de $(Ga\frac{h_0}{r_0})^{\frac{1}{2m+3}}$ pour le régime visqueux, de $(\frac{1}{Bn}\frac{h_0}{r_0})^{\frac{1}{3}}$ pour le régime plastique et de $(\frac{1}{El}\frac{h_0}{r_0})^{\frac{1}{6}}$ pour le régime élastique. 
* L'obtention de droites permet de valider la cohérence du modèle avec la littérature, et l'expression de la loi d'échelle élastique.

CourbeMaitresse_Tau0.py :
* Utilise la méthode solve_ivp de scipy.integrate pour calculer le rayon final pour différentes valeurs de $\tau_0$.
* Trace $\frac{r_\infty}{r_0}/(Ga\frac{h_0}{r_0})^{\frac{1}{2m+3}}$ en fonction de $(\frac{1}{Bn}\frac{h_0}{r_0})^{\frac{1}{3}}/(Ga\frac{h_0}{r_0})^{\frac{1}{2m+3}}$ pour visualiser les deux régimes et vérifier les lois d'échelle.
* Les simulations faites dans ce code sont stockées dans le tableau excel Simulations.xlsx

CourbeMaitresse_Tau0_GlissAdh.py :
* Comme CourbeMaitresse_Tau0.py, avec le cas de glissement en plus.

### Comparaison avec des données expérimentales, issues de la littérature

ExpData_Df.py :
* Récupère les données expérimentales stockées dans le tableau Excel dambreak_experimental_data.xlsx
* Les stocke dans un DataFrame

modeleKV_ExpData.py :
* Calcule les données finales pour chaque expérience.
* Trace la hauteur normalisée calculée par le modèle et celle donnée par l'expérience, pour chaque expérience (et matériau), en fonction du rapport de vitesse caractéristique $u_c$ par la vitesse de transition de régime $u_{crossover}$.
* Trace l'erreur entre la hauteur normalisée expérimentale et celle donnée par le modèle, en fonction de $u_c/u_{crossover}$.

### Comparaison avec nos données expérimentales

Comp_ExpCMod.py :
* Trace le rayon d'étalement final expérimental en fonction du rayon d'étalement final calculé par le modèle.
* Le modèle correspond parfaitement à l'expérience si tous les points sont répartis sur la droite x = y. 

## Modèle de Maxwell

modeleM.py :
* Résoud le système avec une méthode Euler explicite.
* Trace le rayon, la vitesse et la déformation en fonction du temps.
* Diverge.

modeleM2.py :
* Version régularisée du modèle précédent, pour essayer de diminuer la divergence.
* Diverge quand même.