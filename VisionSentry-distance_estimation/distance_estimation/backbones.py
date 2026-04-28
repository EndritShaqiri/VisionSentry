from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import numpy as np

from distance_estimation.types import CameraMetadata


class DepthBackboneResult:
    def __init__(
        self,
        *,
        depth_map_m: np.ndarray | None = None,
        confidence_map: np.ndarray | None = None,
        backbone_name: str = "none",
        warning: str | None = None,
    ) -> None:
        self.depth_map_m = depth_map_m
        self.confidence_map = confidence_map
        self.backbone_name = backbone_name
        self.warning = warning


class BaseDepthBackbone(ABC):
    name = "base"

    @abstractmethod
    def estimate(
        self,
        frame_bgr: np.ndarray,
        *,
        modality: str,
        camera_metadata: CameraMetadata,
    ) -> DepthBackboneResult:
        raise NotImplementedError


class NoOpDepthBackbone(BaseDepthBackbone):
    name = "none"

    def __init__(self, warning: str | None = None) -> None:
        self.warning = warning

    def estimate(
        self,
        frame_bgr: np.ndarray,
        *,
        modality: str,
        camera_metadata: CameraMetadata,
    ) -> DepthBackboneResult:
        return DepthBackboneResult(backbone_name=self.name, warning=self.warning)


class ExternalPythonDepthBackbone(BaseDepthBackbone):
    name = "python"

    def __init__(self, module_name: str, class_name: str, init_kwargs: dict[str, Any] | None = None) -> None:
        module = import_module(module_name)
        backbone_cls = getattr(module, class_name)
        self.backend = backbone_cls(**(init_kwargs or {}))

    def estimate(
        self,
        frame_bgr: np.ndarray,
        *,
        modality: str,
        camera_metadata: CameraMetadata,
    ) -> DepthBackboneResult:
        output = self.backend.predict(frame_bgr, modality=modality, camera_metadata=camera_metadata)
        return DepthBackboneResult(
            depth_map_m=output.get("depth_map_m"),
            confidence_map=output.get("confidence_map"),
            backbone_name=output.get("backbone_name", self.name),
            warning=output.get("warning"),
        )


def build_depth_backbone(cfg: dict[str, Any]) -> BaseDepthBackbone:
    name = str(cfg.get("name", "none")).lower()
    allow_missing = bool(cfg.get("allow_missing", True))
    custom_cfg = cfg.get("custom", {})

    if name in {"none", "off", "heuristic"}:
        return NoOpDepthBackbone()

    if name == "python":
        module_name = custom_cfg.get("module")
        class_name = custom_cfg.get("class")
        if not module_name or not class_name:
            message = "Custom python depth backbone requires custom.module and custom.class."
            if allow_missing:
                return NoOpDepthBackbone(warning=message)
            raise ValueError(message)
        return ExternalPythonDepthBackbone(
            module_name=str(module_name),
            class_name=str(class_name),
            init_kwargs=custom_cfg.get("init_kwargs", {}),
        )

    message = (
        f"Depth backbone '{name}' is not bundled in this repo. "
        "Use depth_backbone.name=python with a custom wrapper, or leave it as none."
    )
    if allow_missing:
        return NoOpDepthBackbone(warning=message)
    raise ValueError(message)
