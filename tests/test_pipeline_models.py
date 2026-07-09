"""Tests para DTOs de ejecución del pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import get_args

from tono_politico.pipeline.models import (
    PhaseName,
    PhaseRunStatus,
    RunManifest,
    RunResult,
    RunStatus,
    VideoRunStatus,
)


class TestVideoRunStatus:
    def test_defaults_representan_video_pendiente_sin_error(self):
        status = VideoRunStatus(video_id="abc123")

        assert is_dataclass(status)
        assert status.video_id == "abc123"
        assert status.titulo == ""
        assert status.descargado is False
        assert status.transcrito is False
        assert status.diarizado is False
        assert status.segmentos_actor == 0
        assert status.omitido is False
        assert status.error is None

    def test_asdict_es_serializable_con_primitivos(self):
        status = VideoRunStatus(
            video_id="abc123",
            titulo="Entrevista",
            descargado=True,
            transcrito=True,
            diarizado=False,
            segmentos_actor=3,
            omitido=True,
            error="descarga fallida",
        )

        assert asdict(status) == {
            "video_id": "abc123",
            "titulo": "Entrevista",
            "descargado": True,
            "transcrito": True,
            "diarizado": False,
            "segmentos_actor": 3,
            "omitido": True,
            "error": "descarga fallida",
        }


class TestPhaseRunStatus:
    def test_phase_defaults_representan_fase_ok_sin_mensaje(self):
        status = PhaseRunStatus(phase="ingesta", ok=True)

        assert status.phase == "ingesta"
        assert status.ok is True
        assert status.elapsed_seconds == 0.0
        assert status.message == ""

    def test_literal_phase_name_documenta_fases_validas(self):
        assert get_args(PhaseName) == (
            "ingesta",
            "diarizacion",
            "segmentacion",
            "temas",
            "filtrado",
            "tono",
            "salida",
            "speech2text",
            "argument_shape",
            "topics_cluster",
            "topics_approach",
        )


class TestRunManifest:
    def test_defaults_representan_manifest_sin_videos_ni_fases(self):
        manifest = RunManifest(
            run_id="run-001",
            playlist_url="https://youtube.com/playlist?list=x",
            playlist_name="Play-PoliTest",
            status="ok",
        )

        assert manifest.run_id == "run-001"
        assert manifest.playlist_url == "https://youtube.com/playlist?list=x"
        assert manifest.playlist_name == "Play-PoliTest"
        assert manifest.status == "ok"
        assert manifest.videos == []
        assert manifest.phases == []
        assert manifest.artifacts_dir is None
        assert manifest.cache_dir is None

    def test_mutables_no_comparten_estado_entre_manifests(self):
        first = RunManifest("run-001", "url", "playlist", "ok")
        second = RunManifest("run-002", "url", "playlist", "ok")

        first.videos.append(VideoRunStatus(video_id="abc123"))
        first.phases.append(PhaseRunStatus(phase="ingesta", ok=True))

        assert len(first.videos) == 1
        assert len(first.phases) == 1
        assert second.videos == []
        assert second.phases == []

    def test_asdict_conserva_paths_para_serializador_explicito_posterior(self):
        manifest = RunManifest(
            run_id="run-001",
            playlist_url="url",
            playlist_name="playlist",
            status="partial",
            videos=[VideoRunStatus(video_id="abc123", error="HTTP 403")],
            phases=[PhaseRunStatus(phase="ingesta", ok=False, message="parcial")],
            artifacts_dir=Path("output/run-001"),
            cache_dir=Path("data/Play-PoliTest"),
        )

        assert asdict(manifest) == {
            "run_id": "run-001",
            "playlist_url": "url",
            "playlist_name": "playlist",
            "status": "partial",
            "videos": [
                {
                    "video_id": "abc123",
                    "titulo": "",
                    "descargado": False,
                    "transcrito": False,
                    "diarizado": False,
                    "segmentos_actor": 0,
                    "omitido": False,
                    "error": "HTTP 403",
                }
            ],
            "phases": [
                {
                    "phase": "ingesta",
                    "ok": False,
                    "elapsed_seconds": 0.0,
                    "message": "parcial",
                }
            ],
            "artifacts_dir": Path("output/run-001"),
            "cache_dir": Path("data/Play-PoliTest"),
        }

    def test_literal_run_status_documenta_estados_validos(self):
        assert get_args(RunStatus) == ("ok", "partial", "failed")


class TestRunResult:
    def test_defaults_representan_ejecucion_exitosa_sin_informe(self):
        manifest = RunManifest("run-001", "url", "playlist", "ok")
        result = RunResult(manifest=manifest)

        assert result.manifest is manifest
        assert result.exit_code == 0
        assert result.informe_path is None

    def test_asdict_incluye_manifest_y_ruta_de_informe(self):
        result = RunResult(
            manifest=RunManifest("run-001", "url", "playlist", "ok"),
            exit_code=0,
            informe_path=Path("output/informe.json"),
        )

        assert asdict(result) == {
            "manifest": {
                "run_id": "run-001",
                "playlist_url": "url",
                "playlist_name": "playlist",
                "status": "ok",
                "videos": [],
                "phases": [],
                "artifacts_dir": None,
                "cache_dir": None,
            },
            "exit_code": 0,
            "informe_path": Path("output/informe.json"),
            "manifest_path": None,
        }
