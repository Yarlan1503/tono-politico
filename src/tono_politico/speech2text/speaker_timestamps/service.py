"""SpeakerTimestampsService — diarización + identificación del actor.

Entrada: AudioVideo (ruta .wav).
Salida: list[TurnoOrador] solo del actor aceptado.

Reutiliza el stack pyannote de ``speech2text.diarization`` durante la
migración; la frontera del paquete es la API por video.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from tono_politico.speech2text.audio_fetcher.models import AudioVideo

from ..diarization.adapter import load_pyannote_pipeline, run_pyannote_pipeline
from ..diarization.matching import identificar_actor
from ..diarization.models import PerfilVozActor, TurnoOrador
from ..diarization.perfil_voz import construir_perfil_desde_output

logger = logging.getLogger(__name__)


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
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None
