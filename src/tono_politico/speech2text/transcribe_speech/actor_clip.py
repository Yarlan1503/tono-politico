"""Construcción de clips actor-only y remapeo al timeline original."""

from __future__ import annotations

from pathlib import Path

from ..models import ActorTranscript, ActorTranscriptSegment, AsrMetadata
from ..speaker_timestamps.models import TurnoOrador
from .models import ClipTranscriber, ClipTranscriptSegment

SCHEMA_VERSION = "actor_transcript.v1"
SCOPE_ACTOR_ONLY = "actor_only"
ASR_PROVIDER = "whisper"


def transcribir_turnos_actor(
    audio_path: Path | str,
    turnos_actor: list[TurnoOrador],
    *,
    video_id: str,
    actor: str,
    transcriptor: ClipTranscriber,
    modelo: str = "large-v3-turbo",
    idioma: str = "es",
    padding_segundos: float = 0.0,
    duracion_audio: float | None = None,
    fecha: str | None = None,
) -> ActorTranscript:
    """Transcribe sólo turnos del actor y conserva source_turn absoluto."""
    ruta_audio = Path(audio_path)
    _validar_entrada(ruta_audio, turnos_actor, video_id, padding_segundos)

    segments: list[ActorTranscriptSegment] = []
    for turno in turnos_actor:
        clip_start, clip_end = _clip_bounds(turno, padding_segundos, duracion_audio)
        clip_segments = transcriptor.transcribir_clip(
            ruta_audio,
            t_start=clip_start,
            t_end=clip_end,
            modelo=modelo,
            idioma=idioma,
        )
        actor_segment = _segmento_actor_desde_clip(turno, clip_start, clip_segments)
        if actor_segment is not None:
            segments.append(actor_segment)

    return ActorTranscript(
        schema_version=SCHEMA_VERSION,
        video_id=video_id,
        actor=actor,
        scope=SCOPE_ACTOR_ONLY,
        asr=AsrMetadata(provider=ASR_PROVIDER, model=modelo, language=idioma),
        segments=segments,
        fecha=fecha,
    )


def _validar_entrada(
    audio_path: Path,
    turnos_actor: list[TurnoOrador],
    video_id: str,
    padding_segundos: float,
) -> None:
    if padding_segundos < 0:
        raise ValueError("padding_segundos no puede ser negativo")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")
    for turno in turnos_actor:
        if turno.video_id != video_id:
            raise ValueError(f"Turno con video_id={turno.video_id!r} no coincide con {video_id!r}")
        if turno.t_start < 0 or turno.t_end <= turno.t_start:
            raise ValueError("Turno inválido: t_end debe ser mayor que t_start")


def _clip_bounds(
    turno: TurnoOrador,
    padding_segundos: float,
    duracion_audio: float | None,
) -> tuple[float, float]:
    clip_start = max(0.0, turno.t_start - padding_segundos)
    clip_end = turno.t_end + padding_segundos
    if duracion_audio is not None:
        clip_end = min(duracion_audio, clip_end)
    if clip_end <= clip_start:
        raise ValueError("ventana de clip inválida después de aplicar padding")
    return clip_start, clip_end


def _segmento_actor_desde_clip(
    turno: TurnoOrador,
    clip_start: float,
    clip_segments: list[ClipTranscriptSegment],
) -> ActorTranscriptSegment | None:
    segmentos_con_texto = [segment for segment in clip_segments if segment.text.strip()]
    if not segmentos_con_texto:
        return None

    text = " ".join(segment.text.strip() for segment in segmentos_con_texto)
    t_start = _clamp(
        clip_start + min(segment.t_start for segment in segmentos_con_texto),
        turno.t_start,
        turno.t_end,
    )
    t_end = _clamp(
        clip_start + max(segment.t_end for segment in segmentos_con_texto),
        turno.t_start,
        turno.t_end,
    )
    return ActorTranscriptSegment(
        text=text,
        t_start=t_start,
        t_end=t_end,
        speaker=turno.speaker_id,
        source_turn_start=turno.t_start,
        source_turn_end=turno.t_end,
        word_count=len(text.split()),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
