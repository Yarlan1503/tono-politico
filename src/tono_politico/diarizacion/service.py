"""Componente 1.5: Diarización e identificación de actor — service OOP.

Orquesta las funciones puras del componente:
    1. Construye el PerfilVozActor una sola vez (cache en memoria)
    2. Por cada video: diariza (con embeddings incluidos) → identifica actor → filtra

Arquitectura basada en WhisperX y pyannote.audio 4.0:
    - El pipeline community-1 ya devuelve speaker_embeddings (256 dims por speaker)
    - No se carga un modelo separado de embeddings de voz
    - Token de HF se pasa al pipeline para modelos gated
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..ingesta.cache import ruta_audio
from ..models import VideoTranscript
from .adapter import load_pyannote_pipeline, run_pyannote_pipeline
from .alineacion import filtrar_por_actor
from .matching import identificar_actor
from .models import TurnoOrador

logger = logging.getLogger(__name__)


class DiarizacionService:
    """Service del Componente 1.5: identifica y filtra al actor objetivo.

    Attributes:
        actor: Nombre del actor político objetivo.
        video_ref_id: ID del video de referencia de voz.
        data_dir: Directorio raíz de datos (mismo que IngestaService).
        umbral_match: Distancia coseno por debajo de la cual se acepta.
        umbral_ambiguo: Distancia por encima de la cual se rechaza.
    """

    def __init__(
        self,
        actor: str = "Lilly Téllez",
        video_ref_id: str = "su9nURIj9XQ",
        data_dir: Path = Path("data"),
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        fallback_pipeline: str | None = "pyannote-community/speaker-diarization-community-1",
        device: str = "auto",
        umbral_match: float = 0.5,
        umbral_ambiguo: float = 0.7,
    ) -> None:
        self.actor = actor
        self.video_ref_id = video_ref_id
        self.data_dir = data_dir
        self.pipeline_name = pipeline_name
        self.fallback_pipeline = fallback_pipeline
        self.device = device
        self.umbral_match = umbral_match
        self.umbral_ambiguo = umbral_ambiguo

        # Lazy-load
        self._pipeline: Any = None
        self._perfil: Any = None

    def procesar(
        self,
        transcripts: list[VideoTranscript],
        nombre_playlist: str,
    ) -> list[VideoTranscript]:
        """Filtra transcripciones conservando solo intervenciones del actor.

        Por cada video: diariza el audio, extrae embeddings por speaker desde
        el pipeline, identifica cuál es el actor, y filtra los SegmentoRaw.

        Args:
            transcripts: Lista de VideoTranscript del Componente 1.
            nombre_playlist: Nombre de la playlist (para resolver rutas de audio).

        Returns:
            Lista de VideoTranscript filtrados (mismo orden, mismo metadata,
            solo segmentos del actor).
        """
        if not transcripts:
            return []

        # 1. Construir perfil de voz una sola vez
        perfil = self._get_perfil(nombre_playlist)

        # 2. Resolver dependencias
        pipeline = self._get_pipeline()

        resultados: list[VideoTranscript] = []

        for transcript in transcripts:
            audio_path = self._audio_path(transcript.video_id, nombre_playlist)

            # 2a. Diarizar + extraer embeddings (una sola llamada al pipeline)
            output = run_pyannote_pipeline(pipeline, str(audio_path))

            turnos = _extraer_turnos(output, transcript.video_id)
            if not turnos:
                logger.info(f"Video {transcript.video_id}: sin turnos diarizados, saltando")
                resultados.append(self._transcript_vacio(transcript))
                continue

            # 2b. Embeddings por speaker (ya calculados por el pipeline)
            speaker_embs = _extraer_embeddings(output)
            if not speaker_embs:
                logger.info(f"Video {transcript.video_id}: sin embeddings de speaker, saltando")
                resultados.append(self._transcript_vacio(transcript))
                continue

            # 2c. Identificar actor
            matches = identificar_actor(
                speaker_embs,
                perfil,
                umbral_match=self.umbral_match,
                umbral_ambiguo=self.umbral_ambiguo,
            )
            speakers_actor = [m.speaker_id for m in matches if m.aceptado]

            if not speakers_actor:
                logger.info(f"Video {transcript.video_id}: actor no identificado, 0 segmentos")
                resultados.append(self._transcript_vacio(transcript))
                continue

            # 2d. Filtrar por turnos del actor
            turnos_actor = [t for t in turnos if t.speaker_id in speakers_actor]
            filtrado = filtrar_por_actor(transcript, turnos_actor)
            resultados.append(filtrado)

        total_seg = sum(len(r.raw_segments) for r in resultados)
        logger.info(
            f"Diarización completa: {total_seg} segmentos del actor "
            f"'{self.actor}' de {len(transcripts)} videos"
        )
        return resultados

    # ──────────────────────────────────────────────────────
    # Resolvedores de ruta
    # ──────────────────────────────────────────────────────

    def _audio_path(self, video_id: str, nombre_playlist: str = "") -> Path:
        """Resuelve la ruta del .wav de un video."""
        return ruta_audio(nombre_playlist, video_id, self.data_dir)

    def _ref_audio_path(self, nombre_playlist: str) -> Path:
        """Resuelve la ruta del .wav de referencia del actor."""
        return ruta_audio(nombre_playlist, self.video_ref_id, self.data_dir)

    # ──────────────────────────────────────────────────────
    # Lazy-load de modelos
    # ──────────────────────────────────────────────────────

    def _get_pipeline(self) -> Any:
        """Carga perezosa del pipeline de diarización pyannote.

        El pipeline community-1 incluye internamente el modelo de embedding,
        por lo que no se carga un modelo separado de embeddings de voz.
        """
        if self._pipeline is None:
            token = _leer_token_hf()

            logger.info(f"Cargando pipeline: {self.pipeline_name}")
            loaded = load_pyannote_pipeline(
                primary_pipeline=self.pipeline_name,
                fallback_pipeline=self.fallback_pipeline,
                token=token,
                device=self.device,
            )
            self._pipeline = loaded.pipeline
            self.pipeline_name = loaded.pipeline_name
        return self._pipeline

    def _get_perfil(self, nombre_playlist: str) -> Any:
        """Construye el perfil de voz del actor (una sola vez, cache en memoria).

        Ejecuta el pipeline sobre el audio de referencia y construye el perfil
        desde output.speaker_embeddings público (sin acceder a _inferences).
        """
        if self._perfil is None:
            pipeline = self._get_pipeline()
            ref_path = self._ref_audio_path(nombre_playlist)

            output = run_pyannote_pipeline(pipeline, str(ref_path))

            from .perfil_voz import construir_perfil_desde_output

            self._perfil = construir_perfil_desde_output(
                output,
                actor=self.actor,
                video_ref_id=self.video_ref_id,
                pipeline_name=self.pipeline_name,
            )
            logger.info(
                f"Perfil de voz construido: actor='{self.actor}', "
                f"dim={len(self._perfil.embedding)}, "
                f"modelo={self._perfil.modelo_embedding}"
            )
        return self._perfil

    # ──────────────────────────────────────────────────────
    # Helpers internos
    # ──────────────────────────────────────────────────────

    def _transcript_vacio(self, transcript: VideoTranscript) -> VideoTranscript:
        """Devuelve un VideoTranscript con metadata igual pero sin segments."""
        return VideoTranscript(
            video_id=transcript.video_id,
            url=transcript.url,
            titulo=transcript.titulo,
            fecha=transcript.fecha,
            raw_segments=[],
        )


# ──────────────────────────────────────────────────────────
# Funciones puras de extracción desde el output del pipeline
# ──────────────────────────────────────────────────────────


def _extraer_turnos(output: Any, video_id: str) -> list[TurnoOrador]:
    """Extrae TurnoOrador desde exclusive_speaker_diarization."""
    turnos: list[TurnoOrador] = []
    for segment, _track, speaker in output.exclusive_speaker_diarization.itertracks(
        yield_label=True
    ):
        turnos.append(
            TurnoOrador(
                video_id=video_id,
                speaker_id=speaker,
                t_start=segment.start,
                t_end=segment.end,
            )
        )
    return turnos


def _extraer_embeddings(output: Any) -> dict[str, list[float]]:
    """Extrae embeddings por speaker desde output.speaker_embeddings.

    El pipeline community-1 devuelve un array numpy de (n_speakers, 256)
    alineado con diarization.labels().
    """
    import numpy as np

    labels = list(output.speaker_diarization.labels())
    embs = output.speaker_embeddings

    if embs is None or len(embs) == 0:
        return {}

    result: dict[str, list[float]] = {}
    for i, label in enumerate(labels):
        emb = np.array(embs[i]).flatten()
        result[label] = emb.tolist()

    return result


def _leer_token_hf() -> str | None:
    """Lee el token de Hugging Face desde archivo cache o env var."""
    import os

    # 1. Env var
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # 2. Archivo cache
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()

    return None
