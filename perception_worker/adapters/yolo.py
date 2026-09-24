import uuid
from pathlib import Path

import numpy as np

from .base import ModelAdapter
from ..schemas import Detection, ModelManifest


def resolution_inference(manifest: ModelManifest) -> tuple[int, int]:
    """Resolution (hauteur, largeur) a laquelle le modele infere.

    Par defaut, celle du manifeste du modele. Une Box peut la surcharger avec
    `config.imgsz = [hauteur, largeur]` : c'est un reglage de deploiement, au
    meme titre que la cadence ou le seuil. Inferer au format natif de la camera
    (480x640 sur OSCAR-02) evite de calculer sur des bandes de remplissage.
    Les dimensions sont ramenees au multiple de 32 inferieur, contrainte des
    reseaux YOLO, et bornees pour qu'une valeur aberrante ne sature pas le CPU.
    """
    defaut = (int(manifest.input.get("height", 640)), int(manifest.input.get("width", 640)))
    brut = (manifest.config or {}).get("imgsz")
    if not isinstance(brut, (list, tuple)) or len(brut) != 2:
        return defaut
    try:
        hauteur, largeur = (max(160, min(1280, int(v) // 32 * 32)) for v in brut)
    except (TypeError, ValueError):
        return defaut
    return hauteur, largeur


class UltralyticsAdapter(ModelAdapter):
    def __init__(self, manifest: ModelManifest, artifact: Path):
        super().__init__(manifest, artifact)
        from ultralytics import YOLO

        self.model = YOLO(str(artifact), task="detect")

    def infer(self, rgb_frame: np.ndarray,
              manifest: ModelManifest | None = None) -> list[Detection]:
        actif = manifest or self.manifest
        height, width = resolution_inference(actif)
        result = self.model.predict(
            source=rgb_frame,
            imgsz=(height, width),
            conf=actif.confidence,
            iou=actif.iou_threshold,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxyn.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names or {}
        detections = []
        for (x1, y1, x2, y2), confidence, class_id in zip(boxes, confidences, classes):
            left = float(max(0.0, min(1.0, x1)))
            top = float(max(0.0, min(1.0, y1)))
            right = float(max(0.0, min(1.0, x2)))
            bottom = float(max(0.0, min(1.0, y2)))
            if right <= left or bottom <= top:
                continue
            detections.append(Detection(
                detection_id=uuid.uuid4().hex,
                label=str(names.get(int(class_id), class_id)),
                class_id=int(class_id),
                confidence=float(confidence),
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            ))
        return detections
