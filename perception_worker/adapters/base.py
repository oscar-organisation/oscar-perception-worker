from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..schemas import Detection, ModelManifest


class ModelAdapter(ABC):
    def __init__(self, manifest: ModelManifest, artifact: Path):
        self.manifest = manifest
        self.artifact = artifact

    @abstractmethod
    def infer(self, rgb_frame: np.ndarray,
              manifest: ModelManifest | None = None) -> list[Detection]:
        """Return normalized top-left XYWH detections."""
