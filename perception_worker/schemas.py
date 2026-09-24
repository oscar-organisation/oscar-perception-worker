from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field, field_validator, model_validator

# Un datagramme sans fragmentation sur un chemin MTU 1 500 classique.
LOSSY_MAX_BYTES = 1200
# Sous la limite de 15 Kio d'un message de donnees LiveKit, en-tetes compris.
RELIABLE_MAX_BYTES = 14_000


class ModelManifest(BaseModel):
    id: str
    name: str
    version: str
    task: str
    runtime: str
    sha256: str
    artifact_name: str
    artifact_path: str
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    labels: list[str] = Field(default_factory=list)
    inference_fps: int = Field(default=5, ge=1, le=30)
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    overlay_enabled: bool = True
    incident_enabled: bool = False
    config: dict = Field(default_factory=dict)


class RuntimeManifest(BaseModel):
    schema_version: str
    robot_id: str
    room: str
    overlay_topic: str = "oscar.vision.overlay"
    models: list[ModelManifest] = Field(default_factory=list)


class WorkerSession(BaseModel):
    livekit_url: str
    room: str
    identity: str
    token: str
    expires_in_seconds: int


class WorkerLeases(BaseModel):
    worker_id: str
    robots: list[str] = Field(default_factory=list)
    renouveler_dans: int = Field(ge=1)


class Detection(BaseModel):
    detection_id: str
    label: str
    class_id: int
    confidence: float = Field(ge=0, le=1)
    # Renseignes par le second etage d'identification, absents sinon. Le
    # visualiseur lit `label` ; ces champs servent aux usages metier.
    product_code: str | None = None
    brand: str | None = None
    category: str | None = None
    match_score: float | None = None
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)

    @field_validator("width", "height")
    @classmethod
    def non_zero_extent(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("une boîte doit avoir une taille positive")
        return value

    @model_validator(mode="after")
    def contained_in_frame(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("une boîte doit rester dans les coordonnées normalisées de l'image")
        return self


class OverlayPacket(BaseModel):
    schema_version: str = Field(default="oscar.vision.overlay.v1", alias="schema")
    robot_id: str
    room: str
    frame_timestamp_us: int
    frame_width: int
    frame_height: int
    model_id: str
    model_name: str
    model_version: str
    task: str
    detections: list[Detection]
    detections_total: int | None = None

    def wire_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")

    def compact_wire_bytes(self) -> bytes:
        """Meme contrat, sans octets inutiles.

        Un identifiant UUID complet et des flottants a seize decimales portaient
        une detection a ~190 octets : dans un paquet lossy de 1 200 octets, il
        n'en tenait que cinq ou six. Un modele de rayon en produit ~90 par image.
        Les noms de champs ne changent pas — l'app de visualisation lit les memes
        cles. Quatre decimales valent 0,06 px sur une image de 640 px.
        L'identifiant reste unique entre modeles : le visualiseur fusionne les
        detections de plusieurs modeles et indexe les boites par identifiant.
        """
        # Empreinte de l'identifiant complet : deux modeles aux identifiants
        # proches (meme debut) ne partagent pas pour autant de prefixe.
        prefixe = hashlib.sha1((self.model_id or "m").encode()).hexdigest()[:6]
        corps = self.model_dump(by_alias=True, exclude_none=True, exclude={"detections"})
        corps["detections"] = [
            {
                "detection_id": f"{prefixe}-{i}",
                "label": d.label,
                "class_id": d.class_id,
                "confidence": round(d.confidence, 3),
                "x": round(d.x, 4), "y": round(d.y, 4),
                "width": round(d.width, 4), "height": round(d.height, 4),
                **({"product_code": d.product_code, "brand": d.brand, "category": d.category,
                    "match_score": round(d.match_score, 3)} if d.product_code else {}),
            }
            for i, d in enumerate(self.detections)
        ]
        return json.dumps(corps, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def fit_wire(self) -> tuple[bytes, bool]:
        """Choisit le canal selon la taille : (octets, fiable ?).

        Jusqu'a LOSSY_MAX_BYTES, canal lossy : un paquet perdu est remplace par
        le suivant 200 ms plus tard, inutile de le retransmettre. Au-dela, le
        canal fiable, borne sous la limite de 15 Kio des messages de donnees
        LiveKit. Si meme cela deborde, on garde les detections les plus sures et
        `detections_total` dit combien il y en avait.
        """
        packet = self.model_copy(deep=True)
        packet.detections.sort(key=lambda item: item.confidence, reverse=True)
        packet.detections_total = len(packet.detections)
        payload = packet.compact_wire_bytes()
        if len(payload) <= LOSSY_MAX_BYTES:
            return payload, False
        while len(payload) > RELIABLE_MAX_BYTES and packet.detections:
            # Retrait par tranche : un retrait unitaire reserialiserait des
            # centaines de fois sur une image dense.
            del packet.detections[max(1, len(packet.detections) * 9 // 10):]
            payload = packet.compact_wire_bytes()
        if len(payload) > RELIABLE_MAX_BYTES:
            raise ValueError("les métadonnées de l'overlay dépassent la taille d'un message de données")
        return payload, True

    def lossy_wire_bytes(self, max_bytes: int = LOSSY_MAX_BYTES) -> bytes:
        """Fit the highest-confidence boxes inside one network MTU."""
        packet = self.model_copy(deep=True)
        packet.detections.sort(key=lambda item: item.confidence, reverse=True)
        packet.detections_total = len(packet.detections)
        payload = packet.compact_wire_bytes()
        while len(payload) > max_bytes and packet.detections:
            packet.detections.pop()
            payload = packet.compact_wire_bytes()
        if len(payload) > max_bytes:
            raise ValueError("les métadonnées de l'overlay dépassent la taille d'un paquet lossy")
        return payload
