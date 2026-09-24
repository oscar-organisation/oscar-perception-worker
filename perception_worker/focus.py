"""Selection des detections « ciblees », facon verrouillage de cible.

Un modele de rayon renvoie ~80 produits par image : tout afficher recouvre la
vue. On ne garde que les quelques objets sur lesquels la camera se porte, ceux
qui sont a la fois proches et pres de l'axe de visee.

Critere retenu apres comparaison visuelle sur une video de rayon :
    score = aire × (1 − distance au centre)²
La taille seule choisissait les produits coupes aux bords de l'image ; le
carre sur la distance regroupe la selection autour du point vise, en
favorisant les plus proches a l'interieur de cette zone.

Les identifiants de detection changent a chaque image : la stabilite passe par
le recouvrement geometrique avec la selection precedente. Sans ce bonus, deux
candidats de score voisin alternaient d'une image a l'autre et la vue clignotait.
"""

from __future__ import annotations

import math

from .schemas import Detection

BONUS_CONTINUITE = 1.5
IOU_CONTINUITE = 0.3


def _iou(a: Detection, b: Detection) -> float:
    largeur = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    hauteur = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    commun = max(0.0, largeur) * max(0.0, hauteur)
    union = a.width * a.height + b.width * b.height - commun
    return commun / union if union > 0 else 0.0


def score_focus(d: Detection) -> float:
    cx, cy = d.x + d.width / 2, d.y + d.height / 2
    distance = math.hypot(cx - 0.5, cy - 0.5) / math.hypot(0.5, 0.5)
    return d.width * d.height * (1.0 - min(1.0, distance)) ** 2


def selectionner(detections: list[Detection], nombre: int,
                 precedentes: list[Detection] | None = None) -> list[Detection]:
    if nombre <= 0 or len(detections) <= nombre:
        return detections
    precedentes = precedentes or []

    def score(d: Detection) -> float:
        base = score_focus(d)
        if any(_iou(d, p) >= IOU_CONTINUITE for p in precedentes):
            base *= BONUS_CONTINUITE
        return base

    return sorted(detections, key=score, reverse=True)[:nombre]


def nombre_cible(config: dict | None) -> int:
    """Lit `config.focus_max` d'un element de Box ; 0 = tout afficher."""
    brut = (config or {}).get("focus_max")
    try:
        return max(0, min(50, int(brut)))
    except (TypeError, ValueError):
        return 0
