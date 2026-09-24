from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import socket


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Variable requise absente: {name}")
    return value


def _mapping(name: str) -> dict[str, str]:
    brut = os.getenv(name, "").strip()
    if not brut:
        return {}
    try:
        valeur = json.loads(brut)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Variable JSON invalide: {name}") from exc
    if not isinstance(valeur, dict) or not all(
        isinstance(cle, str) and isinstance(zones, str)
        for cle, zones in valeur.items()
    ):
        raise RuntimeError(f"Variable attendue comme objet chaîne vers chaîne: {name}")
    return valeur


@dataclass(frozen=True)
class WorkerConfig:
    central_api_url: str
    worker_api_key: str
    worker_id: str
    worker_capacity: int
    livekit_url: str | None
    model_cache: Path
    manifest_refresh_seconds: int = 10
    exclusion_zones: str = ""
    exclusion_overlap: float = 0.6
    exclusion_zones_by_robot: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        return cls(
            central_api_url=_required("OSCAR_CENTRAL_API_URL").rstrip("/"),
            worker_api_key=_required("OSCAR_PERCEPTION_WORKER_KEY"),
            worker_id=os.getenv("OSCAR_WORKER_ID", "").strip() or socket.gethostname(),
            worker_capacity=max(1, min(64, int(os.getenv("OSCAR_WORKER_CAPACITY", "4")))),
            livekit_url=os.getenv("OSCAR_LIVEKIT_URL", "").strip() or None,
            model_cache=Path(os.getenv("OSCAR_MODEL_CACHE", "/var/lib/oscar/models")),
            manifest_refresh_seconds=max(3, int(os.getenv("OSCAR_MANIFEST_REFRESH_SECONDS", "10"))),
            exclusion_zones=os.getenv("OSCAR_EXCLUSION_ZONES", ""),
            exclusion_overlap=min(1.0, max(0.05, float(os.getenv("OSCAR_EXCLUSION_OVERLAP", "0.6")))),
            exclusion_zones_by_robot=_mapping("OSCAR_EXCLUSION_ZONES_BY_ROBOT"),
        )
