from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from .config import WorkerConfig
from .schemas import ModelManifest, RuntimeManifest, WorkerSession


class ModelRegistryClient:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.headers = {"X-OSCAR-Worker-Key": config.worker_api_key}
        self.client = httpx.AsyncClient(base_url=config.central_api_url, headers=self.headers, timeout=60)
        config.model_cache.mkdir(parents=True, exist_ok=True)

    async def manifest(self) -> RuntimeManifest:
        response = await self.client.get(f"/api/ai/runtime/robots/{self.config.robot_id}/manifest")
        response.raise_for_status()
        return RuntimeManifest.model_validate(response.json())

    async def session(self) -> WorkerSession:
        response = await self.client.get(f"/api/ai/runtime/robots/{self.config.robot_id}/session")
        response.raise_for_status()
        return WorkerSession.model_validate(response.json())

    async def artifact(self, model: ModelManifest) -> Path:
        suffix = Path(model.artifact_name).suffix or ".model"
        target = self.config.model_cache / f"{model.id}-{model.sha256[:12]}{suffix}"
        if target.is_file() and self._sha256(target) == model.sha256:
            return target
        partial = target.with_suffix(target.suffix + ".partial")
        async with self.client.stream("GET", model.artifact_path) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                async for chunk in response.aiter_bytes():
                    output.write(chunk)
        if self._sha256(partial) != model.sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(f"Empreinte invalide pour le modèle {model.id}")
        partial.replace(target)
        return target

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
