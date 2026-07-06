"""Componente 1.5: Diarización e identificación de actor — service OOP.

Orquesta las funciones puras del componente:
    1. Construye el PerfilVozActor una sola vez (cache en memoria)
    2. Por cada video: diariza → extrae embeddings por speaker → identifica actor → filtra

El service controla el lazy-loading de pyannote Pipeline, el modelo de
embedding, el Audio helper y el extractor de embeddings por speaker.
Las funciones puras reciben dependencias ya construidas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..ingesta.cache import ruta_audio
from ..models import VideoTranscript
from .alineacion import filtrar_por_actor
from .diarizacion import diarizar
from .matching import identificar_actor
from .models import TurnoOrador
from .perfil_voz import construir_perfil

if TYPE_CHECKING:
    pass

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
        embedding_model: str = "pyannote/embedding",
        umbral_match: float = 0.5,
        umbral_ambiguo: float = 0.7,
    ) -> None:
        self.actor = actor
        self.video_ref_id = video_ref_id
        self.data_dir = data_dir
        self.pipeline_name = pipeline_name
        self.embedding_model = embedding_model
        self.umbral_match = umbral_match
        self.umbral_ambiguo = umbral_ambiguo

        # Lazy-load
        self._pipeline: Any = None
        self._embedding_pipeline: Any = None
        self._audio_helper: Any = None
        self._embedding_extractor: Any = None
        self._perfil: Any = None

    def procesar(
        self,
        transcripts: list[VideoTranscript],
        nombre_playlist: str,
    ) -> list[VideoTranscript]:
        """Filtra transcripciones conservando solo intervenciones del actor.

        Por cada video: diariza el audio, extrae embeddings por speaker,
        identifica cuál es el actor, y filtra los SegmentoRaw.

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
        perfil = self._get_perfil()

        # 2. Resolver dependencias
        pipeline = self._get_pipeline()
        extractor = self._get_embedding_extractor()

        resultados: list[VideoTranscript] = []

        for transcript in transcripts:
            audio_path = self._audio_path(transcript.video_id, nombre_playlist)

            # 2a. Diarizar
            turnos = diarizar(audio_path, pipeline, transcript.video_id)
            if not turnos:
                logger.info(
                    f"Video {transcript.video_id}: sin turnos diarizados, "
                    f"saltando"
                )
                resultados.append(self._transcript_vacio(transcript))
                continue

            # 2b. Extraer embeddings por speaker
            speaker_embs = extractor(audio_path, turnos)

            # 2c. Identificar actor
            matches = identificar_actor(
                speaker_embs,
                perfil,
                umbral_match=self.umbral_match,
                umbral_ambiguo=self.umbral_ambiguo,
            )
            speakers_actor = [m.speaker_id for m in matches if m.aceptado]

            if not speakers_actor:
                logger.info(
                    f"Video {transcript.video_id}: actor no identificado, "
                    f"0 segmentos"
                )
                resultados.append(self._transcript_vacio(transcript))
                continue

            # 2d. Filtrar por turnos del actor
            turnos_actor = [
                t for t in turnos if t.speaker_id in speakers_actor
            ]
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

    def _audio_path(
        self, video_id: str, nombre_playlist: str = ""
    ) -> Path:
        """Resuelve la ruta del .wav de un video.

        Si nombre_playlist está vacío, intenta buscar en cualquier subdirectorio.
        """
        return ruta_audio(nombre_playlist, video_id, self.data_dir)

    def _ref_audio_path(self) -> Path:
        """Resuelve la ruta del .wav de referencia del actor."""
        return ruta_audio("Play-PoliTest", self.video_ref_id, self.data_dir)

    # ──────────────────────────────────────────────────────
    # Lazy-load de modelos
    # ──────────────────────────────────────────────────────

    def _get_pipeline(self) -> Any:
        """Carga perezosa del pipeline de diarización pyannote."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]

            logger.info(f"Cargando pipeline: {self.pipeline_name}")
            self._pipeline = Pipeline.from_pretrained(self.pipeline_name)
        return self._pipeline

    def _get_embedding_pipeline(self) -> Any:
        """Carga perezosa del pipeline de embedding."""
        if self._embedding_pipeline is None:
            from pyannote.audio.pipelines import (  # type: ignore[import-not-found]
                SpeakerEmbedding,
            )

            logger.info(f"Cargando modelo embedding: {self.embedding_model}")
            self._embedding_pipeline = SpeakerEmbedding(
                embedding=self.embedding_model,
            )
        return self._embedding_pipeline

    def _get_audio_helper(self) -> Any:
        """Carga perezosa del helper de audio para medir duración."""
        if self._audio_helper is None:
            from pyannote.audio import Audio  # type: ignore[import-not-found]

            self._audio_helper = Audio(sample_rate=16000, mono="downmix")
        return self._audio_helper

    def _get_embedding_extractor(self) -> Any:
        """Carga perezosa del extractor de embeddings por speaker.

        Por defecto usa la implementación interna que combina el Audio helper
        con el pipeline de embedding para extraer un embedding promedio por
        speaker desde sus turnos.
        """
        if self._embedding_extractor is None:
            self._embedding_extractor = _extraer_embeddings_por_speaker
        return self._embedding_extractor

    def _get_perfil(self) -> Any:
        """Construye el perfil de voz del actor (una sola vez, cache en memoria)."""
        if self._perfil is None:
            perfil = construir_perfil(
                audio_ref=self._ref_audio_path(),
                actor=self.actor,
                video_id_ref=self.video_ref_id,
                modelo_embedding=self.embedding_model,
                embedding_pipeline=self._get_embedding_pipeline(),
                audio_helper=self._get_audio_helper(),
            )
            self._perfil = perfil
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
# Extractor de embeddings por speaker
# ──────────────────────────────────────────────────────────


def _extraer_embeddings_por_speaker(
    audio_path: Path | str,
    turnos: list[TurnoOrador],
) -> dict[str, list[float]]:
    """Extrae el embedding promedio de cada speaker desde sus turnos.

    Para cada speaker_id en los turnos:
        1. Recorta el waveform del audio por cada turno del speaker.
        2. Extrae el embedding de cada fragmento.
        3. Promedia los embeddings en un vector por speaker.

    Usa el modelo de embedding y el Audio helper cargados por el service.

    Args:
        audio_path: Ruta al .wav del video.
        turnos: Lista de TurnoOrador del video (todos los speakers).

    Returns:
        {speaker_id: embedding_promedio} para cada speaker.
    """
    import numpy as np
    from pyannote.audio import Audio  # type: ignore[import-not-found]
    from pyannote.core import Segment  # type: ignore[import-not-found]

    audio_helper = Audio(sample_rate=16000, mono="downmix")
    embedding_model = _get_global_embedding_model()

    # Agrupar turnos por speaker
    turnos_por_speaker: dict[str, list[TurnoOrador]] = {}
    for t in turnos:
        turnos_por_speaker.setdefault(t.speaker_id, []).append(t)

    embeddings: dict[str, list[float]] = {}

    for speaker_id, speaker_turnos in turnos_por_speaker.items():
        embs_turno = []
        for turno in speaker_turnos:
            segment = Segment(turno.t_start, turno.t_end)
            waveform, _ = audio_helper.crop(str(audio_path), segment)
            emb = embedding_model(waveform[None])
            embs_turno.append(np.array(emb))

        # Promedio
        promedio = np.mean(embs_turno, axis=0)
        embeddings[speaker_id] = promedio.flatten().tolist()

    return embeddings


# Estado temporal para el modelo de embedding del extractor
_embedding_model_cache = None


def _get_global_embedding_model():
    """Obtiene el modelo de embedding cacheado para el extractor."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        from pyannote.audio.pipelines import (  # type: ignore[import-not-found]
            SpeakerEmbedding,
        )

        _embedding_model_cache = SpeakerEmbedding(
            embedding="pyannote/embedding",
        )
    return _embedding_model_cache
