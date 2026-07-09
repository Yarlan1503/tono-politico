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
