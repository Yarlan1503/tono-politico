"""DTOs del Componente 1.5: Diarización e identificación de actor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnoOrador:
    """Turno individual de un orador detectado por pyannote.

    Atributos:
        video_id: ID del video de YouTube al que pertenece el turno.
        speaker_id: Etiqueta asignada por pyannote (SPEAKER_00, SPEAKER_01, ...).
        t_start: Tiempo de inicio del turno en segundos.
        t_end: Tiempo de fin del turno en segundos.
    """

    video_id: str
    speaker_id: str
    t_start: float
    t_end: float


@dataclass
class PerfilVozActor:
    """Perfil de voz del actor objetivo, cacheado en memoria durante la ejecución.

    Se construye una sola vez desde un audio de referencia y se compara
    contra cada speaker diarizado mediante distancia coseno.

    Atributos:
        actor: Nombre del actor político objetivo.
        video_id_referencia: ID del video usado como referencia de voz.
        embedding: Embedding promedio del audio de referencia.
        modelo_embedding: Modelo usado para extraer el embedding.
        duracion_segundos: Duración del audio de referencia procesado.
    """

    actor: str
    video_id_referencia: str
    embedding: list[float]
    modelo_embedding: str
    duracion_segundos: float


@dataclass
class SpeakerMatch:
    """Resultado de comparar un speaker diarizado contra el perfil del actor.

    Atributos:
        speaker_id: Etiqueta del speaker evaluado (SPEAKER_00, ...).
        distancia: Distancia coseno entre el speaker y el perfil del actor.
        aceptado: True si el speaker se acepta como el actor objetivo.
        es_ambiguo: True si el match cae en la zona ambigua (descartar).
    """

    speaker_id: str
    distancia: float
    aceptado: bool
    es_ambiguo: bool
