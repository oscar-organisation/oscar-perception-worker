from dataclasses import dataclass
import os
from pathlib import Path
import socket


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Variable requise absente: {name}")
    return value


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
        )
