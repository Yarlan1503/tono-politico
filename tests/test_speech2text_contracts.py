"""Contratos de unidad para speech2text y el control plane."""

from __future__ import annotations

from tono_politico.execution.models import UnitResult
from tono_politico.execution.runner import select_video_metas
from tono_politico.speech2text.audio_fetcher.models import VideoMeta


def _meta(video_id: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        titulo=f"Video {video_id}",
        fecha="20260710",
        duracion=30.0,
    )


def test_unit_result_expresa_estado_y_razon_de_terminalidad() -> None:
    result = UnitResult(
        video_id="vid-1",
        status="skipped",
        reason_code="actor_not_identified",
    )

    assert result.video_id == "vid-1"
    assert result.status == "skipped"
    assert result.reason_code == "actor_not_identified"
    assert result.transcript is None


def test_unit_result_ok_puede_conservar_timings_y_error_opcional() -> None:
    result = UnitResult(
        video_id="vid-1",
        status="ok",
        reason_code="transcript_persisted",
        timings={"download": 1.25, "asr": 2.5},
        error=None,
    )

    assert result.timings == {"download": 1.25, "asr": 2.5}
    assert result.error is None


def test_select_video_metas_filtra_ids_y_aplica_limite_en_orden_original() -> None:
    metas = [_meta("a"), _meta("b"), _meta("c"), _meta("d")]

    selected = select_video_metas(metas, only_video_ids=["d", "b", "missing"], max_videos=1)

    assert [meta.video_id for meta in selected] == ["b"]


def test_select_video_metas_sin_filtros_devuelve_copia() -> None:
    metas = [_meta("a"), _meta("b")]

    selected = select_video_metas(metas)

    assert [meta.video_id for meta in selected] == ["a", "b"]
    assert selected is not metas
