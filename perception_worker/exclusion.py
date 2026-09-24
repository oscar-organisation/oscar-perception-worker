"""Zones de l'image ou une detection n'a pas de sens.

La camera d'un robot voit souvent une partie de lui-meme : pince, pare-chocs,
capot. Ces pieces sont fixes par rapport a la camera et reviennent sur chaque
image, si bien qu'un modele qui s'y trompe une fois s'y trompe en continu —
mesure sur OSCAR-02 : la pince prise pour un « sol sale » sur 6 images sur 8.

Le masque est une propriete du robot, pas du modele : il vit donc dans la
configuration du worker, qui sert un seul robot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import Detection


@dataclass(frozen=True)
class Zone:
    x1: float
    y1: float
    x2: float
    y2: float

    def aire_commune(self, d: Detection) -> float:
        largeur = min(self.x2, d.x + d.width) - max(self.x1, d.x)
        hauteur = min(self.y2, d.y + d.height) - max(self.y1, d.y)
        return max(0.0, largeur) * max(0.0, hauteur)


def lire_zones(brut: str | None) -> tuple[Zone, ...]:
    """Lit « x1,y1,x2,y2;x1,y1,x2,y2 » en coordonnees normalisees.

    Une zone mal formee fait echouer le demarrage : un masque silencieusement
    ignore laisserait revenir les faux positifs sans que personne ne le sache.
    """
    if not brut or not brut.strip():
        return ()
    zones = []
    for morceau in brut.split(";"):
        morceau = morceau.strip()
        if not morceau:
            continue
        valeurs = [float(v) for v in morceau.split(",")]
        if len(valeurs) != 4:
            raise ValueError(f"Zone d'exclusion invalide « {morceau} » : 4 valeurs attendues")
        x1, y1, x2, y2 = valeurs
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Zone d'exclusion invalide « {morceau} » : bornes hors de [0, 1]")
        zones.append(Zone(x1, y1, x2, y2))
    return tuple(zones)


def filtrer(detections: list[Detection], zones: tuple[Zone, ...], recouvrement: float) -> list[Detection]:
    """Retire les detections dont la part situee dans une zone atteint `recouvrement`.

    Le critere porte sur la part de la boite, pas sur son centre : un objet pose
    juste devant les machoires deborde un peu sur la bande de la pince et doit
    rester visible, alors qu'une boite essentiellement dessinee sur la pince
    doit disparaitre.
    """
    if not zones:
        return detections
    gardees = []
    for d in detections:
        aire = d.width * d.height
        dedans = max(z.aire_commune(d) for z in zones)
        if aire > 0 and dedans / aire >= recouvrement:
            continue
        gardees.append(d)
    return gardees
