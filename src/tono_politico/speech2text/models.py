"""DTOs del Componente 1.5: Diarización e identificación de actor."""

from __future__ import annotations

from dataclasses import dataclass

from .speaker_timestamps.models import PerfilVozActor, SpeakerMatch, TurnoOrador

__all__ = [
    "TurnoOrador",
    "AsrMetadata",
    "ActorTranscriptSegment",
    "ActorTranscript",
    "TranscriptSource",
    "PerfilVozActor",
    "SpeakerMatch",
]


@dataclass
class AsrMetadata:
    """Metadatos del motor ASR usado para transcribir turnos del actor.

    Atributos:
        provider: Proveedor/implementación ASR (por ejemplo, whisper).
        model: Modelo ASR concreto (por ejemplo, large-v3-turbo).
        language: Idioma configurado para ASR.
    """

    provider: str
    model: str
    language: str


@dataclass
class ActorTranscriptSegment:
    """Turno del actor transcrito y persistible.

    La unidad persistida es el turno atribuido al actor objetivo, no una
    oración ni un segmento temático. ``source_turn_*`` conserva los límites
    originales de pyannote usados como fuente acústica.
    """

    text: str
    t_start: float
    t_end: float
    speaker: str
    source_turn_start: float
    source_turn_end: float
    word_count: int


@dataclass
class TranscriptSource:
    """Metadata de origen opcional asociada a un transcript."""

    playlist_name: str | None = None
    playlist_id: str | None = None
    playlist_url: str | None = None
    video_title: str | None = None
    video_url: str | None = None
    upload_date: str | None = None
    date_source: str | None = None


@dataclass
class ActorTranscript:
    """Transcripción actor-only de un video.

    No persiste timestamps por palabra, probabilidades por palabra ni datos
    verbose de Whisper. Segmentación temática ocurre en componentes posteriores.

    ``fecha`` (YYYYMMDD) se propaga desde VideoMeta para análisis temporal
    en discursive_approach; None si la metadata no está disponible.
    """

    schema_version: str
    video_id: str
    actor: str
    scope: str
    asr: AsrMetadata
    segments: list[ActorTranscriptSegment]
    fecha: str | None = None
    source: TranscriptSource | None = None
