"""DTOs del componente speaker_timestamps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnoOrador:
    """Turno individual de un orador detectado por pyannote."""

    video_id: str
    speaker_id: str
    t_start: float
    t_end: float


@dataclass
class PerfilVozActor:
    """Perfil de voz del actor objetivo, mantenido en memoria durante la corrida."""

    actor: str
    video_id_referencia: str
    embedding: list[float]
    modelo_embedding: str
    duracion_segundos: float


@dataclass
class SpeakerMatch:
    """Resultado de comparar un speaker diarizado contra el perfil del actor."""

    speaker_id: str
    distancia: float
    aceptado: bool
    es_ambiguo: bool
