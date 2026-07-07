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
        pipeline_name: str = "pyannote-community/speaker-diarization-community-1",
        umbral_match: float = 0.5,
        umbral_ambiguo: float = 0.7,
    ) -> None:
        self.actor = actor
        self.video_ref_id = video_ref_id
        self.data_dir = data_dir
        self.pipeline_name = pipeline_name
        self.umbral_match = umbral_match
        self.umbral_ambiguo = umbral_ambiguo

        # Lazy-load
        self._pipeline: Any = None
        self._audio_helper: Any = None
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
            output = pipeline(str(audio_path))

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
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]

            token = _leer_token_hf()

            logger.info(f"Cargando pipeline: {self.pipeline_name}")
            self._pipeline = Pipeline.from_pretrained(
                self.pipeline_name,
                token=token,
            )
        return self._pipeline

    def _get_audio_helper(self) -> Any:
        """Carga perezosa del helper de audio."""
        if self._audio_helper is None:
            from pyannote.audio import Audio  # type: ignore[import-not-found]

            self._audio_helper = Audio(sample_rate=16000, mono="downmix")
        return self._audio_helper

    def _get_perfil(self, nombre_playlist: str) -> Any:
        """Construye el perfil de voz del actor (una sola vez, cache en memoria).

        Usa el modelo de embedding interno del pipeline para extraer el
        embedding del audio de referencia.
        """
        if self._perfil is None:
            pipeline = self._get_pipeline()
            audio_helper = self._get_audio_helper()
            ref_path = self._ref_audio_path(nombre_playlist)

            # Usar el embedding interno del pipeline
            emb_callable = pipeline._inferences["_embedding"]
            ref_dur = audio_helper.get_duration(str(ref_path))
            from pyannote.core import Segment  # type: ignore[import-not-found]

            waveform, _ = audio_helper.crop(ref_path, Segment(0, ref_dur))
            import numpy as np

            emb = np.array(emb_callable(waveform[None])).flatten()

            from .models import PerfilVozActor

            self._perfil = PerfilVozActor(
                actor=self.actor,
                video_id_referencia=self.video_ref_id,
                embedding=emb.tolist(),
                modelo_embedding="pipeline-internal",
                duracion_segundos=ref_dur,
            )
            logger.info(
                f"Perfil de voz construido: actor='{self.actor}', "
                f"dim={len(emb)}, duración={ref_dur:.1f}s"
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
