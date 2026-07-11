"""SpeakerTimestampsService — diarización + identificación del actor.

Entrada: AudioVideo (ruta .wav).
Salida: list[TurnoOrador] solo del actor aceptado.

Usa el stack pyannote (adapter, matching, perfil_voz) internamente.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tono_politico.speech2text.audio_fetcher.models import AudioVideo

from .matching import identificar_actor
from .models import PerfilVozActor, TurnoOrador
from .perfil_voz import construir_perfil_desde_output

logger = logging.getLogger(__name__)


class PyannotePipelineLoadError(RuntimeError):
    """Error accionable al no poder cargar ningún pipeline pyannote."""


@dataclass(frozen=True)
class LoadedPyannotePipeline:
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
    """Carga el pipeline primary y usa fallback opcional sin filtrar tokens."""
    if pipeline_cls is None:
        pipeline_cls = _import_pipeline_cls()
    primary_error: Exception | None = None

    try:
        pipeline = pipeline_cls.from_pretrained(primary_pipeline, token=token)
        _apply_device(pipeline, device, torch_module)
        return LoadedPyannotePipeline(pipeline, primary_pipeline, False)
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
        return LoadedPyannotePipeline(pipeline, fallback_pipeline, True)
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
    """Ejecuta pyannote usando ProgressHook si está disponible."""
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
        from pyannote.audio.pipelines.utils.hook import (
            ProgressHook,  # type: ignore[import-not-found]
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
    if resolved is not None:
        pipeline.to(resolved)


def _resolve_device(device: str, torch_module: Any) -> Any | None:
    if device == "none":
        return None
    target = "cuda" if device == "auto" and torch_module.cuda.is_available() else device
    if device == "auto":
        target = "cuda" if torch_module.cuda.is_available() else "cpu"
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


class SpeakerTimestampsService:
    """Quién habla cuándo, filtrado al actor objetivo."""

    def __init__(
        self,
        actor: str = "Lilly Téllez",
        video_ref_id: str = "su9nURIj9XQ",
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        fallback_pipeline: str | None = "pyannote-community/speaker-diarization-community-1",
        device: str = "auto",
        umbral_match: float = 0.5,
        umbral_ambiguo: float = 0.7,
    ) -> None:
        self.actor = actor
        self.video_ref_id = video_ref_id
        self.pipeline_name = pipeline_name
        self.fallback_pipeline = fallback_pipeline
        self.device = device
        self.umbral_match = umbral_match
        self.umbral_ambiguo = umbral_ambiguo
        self._pipeline: Any = None
        self._perfil: PerfilVozActor | None = None

    def build_perfil(self, ref_audio: AudioVideo) -> PerfilVozActor:
        """Construye (o reutiliza) el perfil de voz del actor desde un audio ref."""
        if self._perfil is not None:
            return self._perfil

        pipeline = self._get_pipeline()
        output = run_pyannote_pipeline(pipeline, str(ref_audio.audio_path))
        self._perfil = construir_perfil_desde_output(
            output,
            actor=self.actor,
            video_ref_id=ref_audio.video_id,
            pipeline_name=self.pipeline_name,
        )
        logger.info(
            "Perfil de voz construido: actor=%r dim=%s",
            self.actor,
            len(self._perfil.embedding),
        )
        return self._perfil

    def set_perfil(self, perfil: PerfilVozActor) -> None:
        """Inyecta un perfil preconstruido (tests / reuso entre corridas)."""
        self._perfil = perfil

    def procesar_one(self, audio: AudioVideo) -> list[TurnoOrador]:
        """Diariza un audio y devuelve solo turnos del actor.

        Returns:
            Lista vacía si no hay turnos, embeddings o match del actor.
        """
        if self._perfil is None:
            raise RuntimeError(
                "Perfil de voz no construido. Llama build_perfil() o set_perfil() antes."
            )

        pipeline = self._get_pipeline()
        output = run_pyannote_pipeline(pipeline, str(audio.audio_path))

        turnos = _extraer_turnos(output, audio.video_id)
        if not turnos:
            logger.info("Video %s: sin turnos diarizados", audio.video_id)
            return []

        speaker_embs = _extraer_embeddings(output)
        if not speaker_embs:
            logger.info("Video %s: sin embeddings de speaker", audio.video_id)
            return []

        matches = identificar_actor(
            speaker_embs,
            self._perfil,
            umbral_match=self.umbral_match,
            umbral_ambiguo=self.umbral_ambiguo,
        )
        speakers_actor = {m.speaker_id for m in matches if m.aceptado}
        if not speakers_actor:
            logger.info("Video %s: actor no identificado", audio.video_id)
            return []

        turnos_actor = [t for t in turnos if t.speaker_id in speakers_actor]
        logger.info(
            "Video %s: %s turnos del actor '%s'",
            audio.video_id,
            len(turnos_actor),
            self.actor,
        )
        return turnos_actor

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            token = _leer_token_hf()
            logger.info("Cargando pipeline: %s", self.pipeline_name)
            loaded = load_pyannote_pipeline(
                primary_pipeline=self.pipeline_name,
                fallback_pipeline=self.fallback_pipeline,
                token=token,
                device=self.device,
            )
            self._pipeline = loaded.pipeline
            self.pipeline_name = loaded.pipeline_name
        return self._pipeline


def _extraer_turnos(output: Any, video_id: str) -> list[TurnoOrador]:
    turnos: list[TurnoOrador] = []
    diarization = getattr(output, "exclusive_speaker_diarization", None)
    if diarization is None or not hasattr(diarization, "itertracks"):
        raise ValueError("output no contiene exclusive_speaker_diarization iterable")
    for segment, _track, speaker in diarization.itertracks(yield_label=True):
        t_start = float(segment.start)
        t_end = float(segment.end)
        if t_start < 0 or t_end <= t_start:
            raise ValueError("turno inválido: t_end debe ser mayor que t_start")
        if not speaker:
            raise ValueError("turno inválido: speaker vacío")
        turnos.append(
            TurnoOrador(
                video_id=video_id,
                speaker_id=str(speaker),
                t_start=t_start,
                t_end=t_end,
            )
        )
    return turnos


def _extraer_embeddings(output: Any) -> dict[str, list[float]]:
    import numpy as np

    diarization = getattr(output, "speaker_diarization", None)
    if diarization is None or not hasattr(diarization, "labels"):
        raise ValueError("output no contiene speaker_diarization.labels")
    labels = list(diarization.labels())
    embs = getattr(output, "speaker_embeddings", None)
    if embs is None:
        return {}
    if len(embs) != len(labels):
        raise ValueError(
            f"cantidad de embeddings inconsistente con labels: {len(embs)} != {len(labels)}"
        )

    result: dict[str, list[float]] = {}
    for label, raw_embedding in zip(labels, embs, strict=True):
        if raw_embedding is None:
            raise ValueError(f"embedding ausente para speaker {label}")
        emb = np.asarray(raw_embedding, dtype=float).reshape(-1)
        if emb.size == 0:
            raise ValueError(f"embedding vacío para speaker {label}")
        if not np.isfinite(emb).all():
            raise ValueError(f"embedding no finito para speaker {label}")
        result[str(label)] = emb.tolist()
    return result


def _leer_token_hf() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None
