"""RED tests for durable speech2text metadata provenance."""

from __future__ import annotations

import json
from pathlib import Path

from tono_politico.execution.artifacts import resolve_artifacts
from tono_politico.execution.config import load_run_config
from tono_politico.execution.plan import build_execution_plan
from tono_politico.execution.runner import ExecutionFactories, ExecutionRunner
from tono_politico.speech2text.audio_fetcher.models import PlaylistInfo, VideoMeta
from tono_politico.speech2text.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
)


def _config(tmp_path: Path):
    path = tmp_path / "run.yaml"
    path.write_text(
        f"""
schema_version: tono-politico.run.v1
run:
  id: metadata-run
  stages: [speech2text]
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
project:
  data_dir: {tmp_path / "data"}
""",
        encoding="utf-8",
    )
    return load_run_config(path)


def _transcript(video_id: str, fecha: str | None) -> ActorTranscript:
    return ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id=video_id,
        actor="Lilly Téllez",
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
        fecha=fecha,
    )


class MetadataSpeechService:
    def discover(self, _url: str) -> tuple[PlaylistInfo, list[VideoMeta]]:
        return PlaylistInfo(
            nombre="Play-PoliTest",
            nombre_cache="Play-PoliTest",
            playlist_id="PLE9Zk7g9R__M",
            url="https://www.youtube.com/playlist?list=PLE9Zk7g9R__M",
        ), [
            VideoMeta(
                video_id="video-1",
                url="https://youtu.be/video-1",
                titulo="Título recuperable",
                fecha="20260511",
                duracion=10.0,
                fecha_fuente="upload_date",
            )
        ]

    def ensure_perfil(self, playlist: PlaylistInfo, _metas: list[VideoMeta]) -> bool:
        assert playlist.cache_name == "Play-PoliTest"
        return True

    def procesar_one(self, video: VideoMeta, playlist: PlaylistInfo) -> ActorTranscript:
        assert playlist.nombre == "Play-PoliTest"
        return _transcript(video.video_id, video.fecha)


def _factories() -> ExecutionFactories:
    return ExecutionFactories(
        build_speech2text=lambda _cfg: MetadataSpeechService(),
        build_argument_shape=lambda _cfg: None,
        build_topics_cluster=lambda _cfg: None,
        build_topics_approach=lambda _cfg: None,
    )


def test_manifest_conserva_playlist_titulo_y_fecha_por_unidad(tmp_path: Path):
    cfg = _config(tmp_path)
    artifacts = resolve_artifacts(cfg, "metadata-run")
    plan = build_execution_plan(cfg, artifacts)

    result = ExecutionRunner(_factories()).execute(plan)

    assert result.exit_code == 0
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["speech2text"]["playlist"]["playlist_id"] == "PLE9Zk7g9R__M"
    assert manifest["speech2text"]["reference_video"]["video_id"] == "su9nURIj9XQ"
    assert manifest["speech2text"]["reference_video"]["reason_code"] == "reference_profile"
    assert manifest["speech2text"]["reference_video"]["present_in_playlist"] is False
    assert manifest["units"][0]["video_title"] == "Título recuperable"
    assert manifest["units"][0]["fecha"] == "20260511"
    assert manifest["units"][0]["fecha_fuente"] == "upload_date"

    checkpoint = json.loads(
        (artifacts.run_dir / "speech2text" / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["speech2text"] == manifest["speech2text"]

    quality = json.loads(artifacts.speech2text_quality_path.read_text(encoding="utf-8"))
    assert quality["provenance"] == manifest["speech2text"]
    assert quality["videos"][0]["video_title"] == "Título recuperable"
