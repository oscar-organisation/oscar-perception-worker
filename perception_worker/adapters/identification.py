"""Second etage : donner un nom aux produits deja detectes.

Le detecteur de rayon dit « il y a un produit ici ». Cet adaptateur recadre
chaque produit retenu, calcule son empreinte visuelle et la compare a une
galerie de reference construite depuis Open Food Facts (photos d'emballage des
produits vendus en France). Le plus proche donne nom, marque et categorie.

Un nom n'est affiche que si la ressemblance depasse un seuil calibre a la
construction de la galerie, et si le premier candidat devance nettement le
second : un faux nom est pire qu'un « produit » generique.

L'artefact embarque tout (encodeur TorchScript, empreintes, fiches produits) :
le worker n'a besoin d'aucun acces reseau pour identifier.

Encodeur : DINOv2 ViT-S/14, retenu sur banc face a CLIP ViT-B/32 (galerie de
1 000 produits, vues degradees) — produit exact retrouve a 38 / 82 / 96 % pour
des produits de 80 / 150 / 250 px de haut, contre 16 / 74 / 92 %.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schemas import ModelManifest

FORMAT = "oscar.identification.v2"


@dataclass(frozen=True)
class Identite:
    code: str
    nom: str
    marque: str
    categorie: str
    score: float

    @property
    def libelle(self) -> str:
        """Libelle court pour l'etiquette du cockpit."""
        nom = self.nom.strip()
        if self.marque and self.marque.lower() not in nom.lower():
            nom = f"{self.marque} · {nom}"
        return nom if len(nom) <= 40 else nom[:39].rstrip() + "…"


def decider(scores: np.ndarray, indices: np.ndarray, seuil: float, marge: float) -> list[int | None]:
    """Index du produit retenu pour chaque recadrage, ou None.

    `scores` et `indices` : les deux meilleurs candidats par ligne, tries.
    Extrait pour etre teste sans encodeur ni galerie reelle.
    """
    retenus: list[int | None] = []
    for (premier, second), (i, _) in zip(scores, indices):
        retenus.append(int(i) if premier >= seuil and premier - second >= marge else None)
    return retenus


class IdentificationAdapter:
    def __init__(self, manifest: ModelManifest, artifact: Path):
        import io

        import torch
        import torchvision.transforms as T

        self.manifest = manifest
        paquet = torch.load(artifact, map_location="cpu", weights_only=False)
        if paquet.get("format") != FORMAT:
            raise ValueError(f"Artefact d'identification inattendu : {paquet.get('format')!r}")
        encodeur = paquet["encoder"]
        if encodeur.get("type") != "torchscript":
            raise ValueError(f"Encodeur non pris en charge : {encodeur.get('type')!r}")
        self._torch = torch
        self.modele = torch.jit.load(io.BytesIO(encodeur["bytes"]), map_location="cpu").eval()
        taille = int(encodeur["input"])
        # Pretraitement identique a celui de la construction de la galerie :
        # la moindre difference fausse les empreintes sans lever d'erreur.
        self.preprocess = T.Compose([
            T.Resize(taille, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(taille),
            T.ToTensor(), T.Normalize(tuple(encodeur["mean"]), tuple(encodeur["std"])),
        ])
        self.galerie = paquet["embeddings"].float()
        self.produits = paquet["products"]
        calibration = paquet.get("calibration") or {}
        config = manifest.config or {}
        # La Box peut resserrer ou relacher les reglages calibres sans reconstruire la galerie.
        self.seuil = float(config.get("match_threshold", calibration.get("seuil", 0.85)))
        self.marge = float(config.get("match_margin", calibration.get("marge", 0.02)))
        self.hauteur_min = int(config.get("min_crop_height", calibration.get("hauteur_min", 140)))

    def identifier(self, recadrages: list[np.ndarray]) -> list[Identite | None]:
        if not recadrages:
            return []
        from PIL import Image

        torch = self._torch
        with torch.no_grad():
            lot = torch.stack([self.preprocess(Image.fromarray(r)) for r in recadrages])
            f = self.modele(lot)
            f = f / f.norm(dim=-1, keepdim=True)
            top = (f @ self.galerie.T).topk(2, dim=1)
        choix = decider(top.values.numpy(), top.indices.numpy(), self.seuil, self.marge)
        identites: list[Identite | None] = []
        for ligne, i in enumerate(choix):
            if i is None:
                identites.append(None)
                continue
            p = self.produits[i]
            identites.append(Identite(p["code"], p["nom"], p.get("marque", ""),
                                      p.get("categorie", ""), float(top.values[ligne, 0])))
        return identites


def recadrer(frame: np.ndarray, x: float, y: float, w: float, h: float,
             marge: float = 0.08, cote_min: int = 24, hauteur_min: int = 0) -> np.ndarray | None:
    """Recadrage d'un produit, legerement elargi pour garder le bord de l'emballage.

    Renvoie None si le produit est trop petit : moins de `cote_min` pixels de
    cote, ou moins de `hauteur_min` de haut. Mesure sur banc : a 80 px de haut
    le bon produit n'arrive en tete qu'une fois sur trois ; mieux vaut laisser
    « produit » que tirer un nom au hasard.
    """
    hauteur, largeur = frame.shape[:2]
    x1 = max(0, int((x - w * marge) * largeur)); y1 = max(0, int((y - h * marge) * hauteur))
    x2 = min(largeur, int((x + w * (1 + marge)) * largeur)); y2 = min(hauteur, int((y + h * (1 + marge)) * hauteur))
    if x2 - x1 < cote_min or y2 - y1 < max(cote_min, hauteur_min):
        return None
    return np.ascontiguousarray(frame[y1:y2, x1:x2])


def appliquer_identites(detections, memoire, maintenant: float, duree_vie: float = 3.0,
                        iou_min: float = 0.3):
    """Reporte les noms connus sur les boites de l'image courante.

    L'identification tourne plus lentement que la detection et en tache de
    fond : chaque image reprend le nom de la boite identifiee qui la recouvre
    le plus, tant que ce nom a moins de `duree_vie` secondes. Quand la camera
    bouge, le recouvrement chute et le libelle redevient generique jusqu'a la
    prochaine identification, plutot que de coller un nom a un voisin.

    `memoire` : liste de (Detection identifiee, Identite | None, instant).
    """
    from ..focus import _iou

    if not memoire:
        return detections
    nommees = []
    for d in detections:
        meilleur, meilleur_iou = None, iou_min
        for ref, identite, instant in memoire:
            if identite is None or maintenant - instant > duree_vie:
                continue
            recouvrement = _iou(d, ref)
            if recouvrement >= meilleur_iou:
                meilleur, meilleur_iou = identite, recouvrement
        if meilleur is None:
            nommees.append(d)
        else:
            nommees.append(d.model_copy(update={
                "label": meilleur.libelle, "product_code": meilleur.code, "brand": meilleur.marque or None,
                "category": meilleur.categorie or None, "match_score": meilleur.score,
            }))
    return nommees
