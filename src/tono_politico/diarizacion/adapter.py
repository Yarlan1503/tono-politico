"""Adapter pyannote: carga primary/fallback, device y ProgressHook.

Este módulo concentra los detalles runtime de pyannote para que
`DiarizacionService` no conozca namespaces, fallback, hooks ni CUDA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class PyannotePipelineLoadError(RuntimeError):
    """Error accionable al no poder cargar ningún pipeline pyannote."""


@dataclass(frozen=True)
class LoadedPyannotePipeline:
    """Resultado de carga de un pipeline pyannote."""

    pipeline: Any
    pipeline_name: str
    used_fallback: bool = False


def load_pyannote_pipeline(
    primary_pipeline: str,
    fallback_pipeline: str | None = None,
    token: str | None = None,
    device: str = "auto",
    pipeline_cls: Any | None = None,
    torch_module: Any | None = None,
) -> LoadedPyannotePipeline:
    """Carga un pipeline pyannote con fallback opcional y device.

    Args:
        primary_pipeline: Nombre del modelo principal.
        fallback_pipeline: Nombre alternativo si el principal falla.
        token: Token HF opcional; nunca se loguea.
        device: "auto", "cpu", "cuda" o similar.
        pipeline_cls: Inyección para tests; por defecto `pyannote.audio.Pipeline`.
        torch_module: Inyección para tests; por defecto `torch` si está disponible.
    """
    if pipeline_cls is None:
        pipeline_cls = _import_pipeline_cls()
    primary_error: Exception | None = None

    try:
        pipeline = pipeline_cls.from_pretrained(primary_pipeline, token=token)
        _apply_device(pipeline, device, torch_module)
        return LoadedPyannotePipeline(
            pipeline=pipeline,
            pipeline_name=primary_pipeline,
            used_fallback=False,
        )
    except Exception as exc:
        primary_error = exc
        if not fallback_pipeline:
            raise _load_error(primary_pipeline, None, primary_error, None) from primary_error
        logger.warning(
            "No se pudo cargar pipeline pyannote principal '%s': %s",
            primary_pipeline,
            primary_error,
        )

    try:
        pipeline = pipeline_cls.from_pretrained(fallback_pipeline, token=token)
        _apply_device(pipeline, device, torch_module)
        return LoadedPyannotePipeline(
            pipeline=pipeline,
            pipeline_name=fallback_pipeline,
            used_fallback=True,
        )
    except Exception as fallback_error:
        assert primary_error is not None
        raise _load_error(
            primary_pipeline,
            fallback_pipeline,
            primary_error,
            fallback_error,
        ) from fallback_error


def run_pyannote_pipeline(
    pipeline: Any,
    audio_path: str,
    progress_hook_cls: Any | None = "auto",
) -> Any:
    """Ejecuta un pipeline pyannote con ProgressHook si está disponible."""
    if progress_hook_cls == "auto":
        progress_hook_cls = _import_progress_hook_cls()

    if progress_hook_cls is None:
        return pipeline(str(audio_path))

    with progress_hook_cls() as hook:
        return pipeline(str(audio_path), hook=hook)


def _import_pipeline_cls() -> Any:
    from pyannote.audio import Pipeline  # type: ignore[import-not-found]

    return Pipeline


def _import_progress_hook_cls() -> Any | None:
    try:
        from pyannote.audio.pipelines.utils.hook import (  # type: ignore[import-not-found]
            ProgressHook,
        )
    except Exception:
        return None
    return ProgressHook


def _apply_device(pipeline: Any, device: str, torch_module: Any | None) -> None:
    if not hasattr(pipeline, "to"):
        return

    torch_module = torch_module or _import_torch_module()
    if torch_module is None:
        return

    resolved = _resolve_device(device, torch_module)
    if resolved is None:
        return
    pipeline.to(resolved)


def _resolve_device(device: str, torch_module: Any) -> Any | None:
    """Resuelve el device y devuelve torch.device (no str).

    pyannote 4.x requiere torch.device, no str.
    """
    if device == "none":
        return None
    if device == "auto":
        target = "cuda" if torch_module.cuda.is_available() else "cpu"
    else:
        target = device
    return torch_module.device(target)


def _import_torch_module() -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    return torch


def _load_error(
    primary_pipeline: str,
    fallback_pipeline: str | None,
    primary_error: Exception,
    fallback_error: Exception | None,
) -> PyannotePipelineLoadError:
    fallback_msg = (
        f"; fallback '{fallback_pipeline}' falló: {fallback_error}" if fallback_pipeline else ""
    )
    return PyannotePipelineLoadError(
        "No se pudo cargar ningún pipeline de diarización pyannote. "
        f"Principal '{primary_pipeline}' falló: {primary_error}"
        f"{fallback_msg}. Verifica acceso/condiciones de Hugging Face y token local."
    )
