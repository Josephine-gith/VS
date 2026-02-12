# Viscoplastic Squeeze
## Modèle de Kelvin Voigt

* modele.py : Calcule la hauteur, le rayon, la vitesse et la déformation au cours d'un temps d'une compression sous les masses M contenues dans ML. Trace le rayon et la hauteur en fonction du temps, pour ces différentes masses, dans les cas glissant et non-glissant.
* ContributionContraintes.py : Calcule la contribution des contraintes résistantes (plastique, visqueuse). Trace un graphe de répartition des contraintes (entre 0 et 1) dans les cas glissant et non-glissant.
* Comp_ExpTemp.py : Trace les mesures expérimentales d'un étalement au cours du temps (dans l'excel "Mesures Temporelles"), et le calcul du modèle pour ces conditions, avec le cas non-glissant pour la colle et glissant pour la mayonnaise.
* Comp_ExpDiffM.py : Trace le diamètre expérimental à t_final en fonction du diamètre calculé par le modèle, pour les mesures faites avec plusieurs masses et plusieurs tailles de gouttes (dans l'excel "Mesures colle" pour celles sur la colle, en attendant 1h30, ou dans l'excel "Toute Mesures" avec la mayonnaise, en attendant un temps incertain).