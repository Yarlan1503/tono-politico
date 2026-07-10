"""Observabilidad de unidades speech2text en el control plane."""

from __future__ import annotations

import json
from pathlib import Path

from tono_politico.execution.models import UnitResult
from tono_politico.execution.observability import (
    build_quality_report,
    guardar_quality_report,
)
from tono_politico.speech2text.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
)


def _transcript(video_id: str) -> ActorTranscript:
    return ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id=video_id,
        actor="Actor",
        scope="actor_only",
        asr=AsrMetadata(provider="whisper", model="tiny", language="es"),
        segments=[
            ActorTranscriptSegment(
                text="Texto",
                t_start=0.0,
                t_end=1.0,
                speaker="SPEAKER_00",
                source_turn_start=0.0,
                source_turn_end=1.0,
                word_count=1,
            )
        ],
    )


def test_reporte_incluye_ok_skip_y_fallo_por_video() -> None:
    report = build_quality_report(
        [
            UnitResult(
                video_id="ok",
                status="ok",
                reason_code="transcript_persisted",
                transcript=_transcript("ok"),
            ),
            UnitResult(
                video_id="skip",
                status="skipped",
                reason_code="actor_not_identified",
            ),
            UnitResult(
                video_id="fail",
                status="failed",
                reason_code="download_failed",
                error="yt-dlp falló",
            ),
        ]
    )

    assert report.schema_version == "speech2text_quality.v2"
    assert report.total_selected_videos == 3
    assert report.videos_ok == 1
    assert report.videos_skipped == 1
    assert report.videos_failed == 1
    assert [(item.video_id, item.status, item.reason_code) for item in report.videos] == [
        ("ok", "ok", "transcript_persisted"),
        ("skip", "skipped", "actor_not_identified"),
        ("fail", "failed", "download_failed"),
    ]


def test_reporte_persistido_no_duplica_texto(tmp_path: Path) -> None:
    report = build_quality_report(
        [
            UnitResult(
                video_id="ok",
                status="ok",
                reason_code="transcript_persisted",
                transcript=_transcript("ok"),
            )
        ]
    )
    path = guardar_quality_report(report, tmp_path / "quality.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "speech2text_quality.v2"
    assert payload["videos"][0]["segment_count"] == 1
    assert "Texto" not in json.dumps(payload, ensure_ascii=False)
