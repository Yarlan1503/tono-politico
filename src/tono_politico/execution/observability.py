"""Observabilidad de speech2text en el control plane.

Este módulo registra estados de ejecución y métricas de contenido sin mover
lógica de dominio ni duplicar el texto de ``ActorTranscript``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import UnitResult

SCHEMA_VERSION = "speech2text_quality.v2"


@dataclass(frozen=True)
class VideoQualityMetrics:
    """Estado y métricas no textuales de una unidad procesada."""

    video_id: str
    status: str
    reason_code: str
    segment_count: int
    word_count: int
    segments_with_one_word: int
    segments_with_two_or_fewer_words: int
    empty_transcript: bool
    video_title: str | None = None
    fecha: str | None = None
    fecha_fuente: str | None = None


@dataclass(frozen=True)
class Speech2TextQualityReport:
    """Informe agregado de todos los vídeos seleccionados."""

    schema_version: str
    total_selected_videos: int
    videos_ok: int
    videos_skipped: int
    videos_failed: int
    transcripts_without_segments: int
    total_segments: int
    total_words: int
    segments_with_one_word: int
    segments_with_two_or_fewer_words: int
    videos: list[VideoQualityMetrics]
    provenance: dict[str, Any] | None = None


def build_quality_report(
    units: Iterable[UnitResult],
    *,
    provenance: dict[str, Any] | None = None,
) -> Speech2TextQualityReport:
    """Construye métricas para éxito, skip y fallo, sin filtrar segmentos."""
    videos = [_video_metrics(unit) for unit in units]
    return Speech2TextQualityReport(
        schema_version=SCHEMA_VERSION,
        total_selected_videos=len(videos),
        videos_ok=sum(video.status == "ok" for video in videos),
        videos_skipped=sum(video.status == "skipped" for video in videos),
        videos_failed=sum(video.status == "failed" for video in videos),
        transcripts_without_segments=sum(
            video.status == "ok" and video.empty_transcript for video in videos
        ),
        total_segments=sum(video.segment_count for video in videos),
        total_words=sum(video.word_count for video in videos),
        segments_with_one_word=sum(video.segments_with_one_word for video in videos),
        segments_with_two_or_fewer_words=sum(
            video.segments_with_two_or_fewer_words for video in videos
        ),
        videos=videos,
        provenance=provenance,
    )


def guardar_quality_report(report: Speech2TextQualityReport, path: Path | str) -> Path:
    """Persiste el informe como JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(quality_report_to_json(report), encoding="utf-8")
    return path


def quality_report_to_json(report: Speech2TextQualityReport) -> str:
    return json.dumps(quality_report_to_dict(report), indent=2, ensure_ascii=False)


def quality_report_to_dict(report: Speech2TextQualityReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "total_selected_videos": report.total_selected_videos,
        "videos_ok": report.videos_ok,
        "videos_skipped": report.videos_skipped,
        "videos_failed": report.videos_failed,
        "transcripts_without_segments": report.transcripts_without_segments,
        "total_segments": report.total_segments,
        "total_words": report.total_words,
        "segments_with_one_word": report.segments_with_one_word,
        "segments_with_two_or_fewer_words": report.segments_with_two_or_fewer_words,
        "provenance": report.provenance,
        "videos": [
            {
                "video_id": video.video_id,
                "status": video.status,
                "reason_code": video.reason_code,
                "segment_count": video.segment_count,
                "word_count": video.word_count,
                "segments_with_one_word": video.segments_with_one_word,
                "segments_with_two_or_fewer_words": video.segments_with_two_or_fewer_words,
                "empty_transcript": video.empty_transcript,
                "video_title": video.video_title,
                "fecha": video.fecha,
                "fecha_fuente": video.fecha_fuente,
            }
            for video in report.videos
        ],
    }


def _video_metrics(unit: UnitResult) -> VideoQualityMetrics:
    transcript = unit.transcript
    segments = transcript.segments if transcript is not None else []
    word_counts = [segment.word_count for segment in segments]
    return VideoQualityMetrics(
        video_id=unit.video_id,
        status=unit.status,
        reason_code=unit.reason_code,
        segment_count=len(word_counts),
        word_count=sum(word_counts),
        segments_with_one_word=sum(count == 1 for count in word_counts),
        segments_with_two_or_fewer_words=sum(count <= 2 for count in word_counts),
        empty_transcript=not word_counts,
        video_title=unit.video_title,
        fecha=unit.fecha,
        fecha_fuente=unit.fecha_fuente,
    )
