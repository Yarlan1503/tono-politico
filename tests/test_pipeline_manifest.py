"""Tests para persistencia de RunManifest y resumen final."""

from __future__ import annotations

import json
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
from typing import Literal

from tono_politico.pipeline.manifest import (
    guardar_manifest,
    manifest_to_dict,
    manifest_to_json,
    resumen_final,
)
from tono_politico.pipeline.models import (
    PhaseRunStatus,
    RunManifest,
    RunResult,
    VideoRunStatus,
)


def _manifest(
    run_id: str = "20260706-153012",
    playlist_name: str = "Play-PoliTest",
    status: Literal["ok", "partial", "failed"] = "ok",
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        playlist_url="https://youtube.com/playlist?list=PLE9",
        playlist_name=playlist_name,
        status=status,
        videos=[
            VideoRunStatus(
                video_id="v1",
                titulo="Video 1",
                descargado=True,
                transcrito=True,
                diarizado=True,
                segmentos_actor=5,
            ),
            VideoRunStatus(
                video_id="v2",
                titulo="Video 2 (403)",
                descargado=False,
                omitido=True,
                error="HTTP 403",
            ),
        ],
        phases=[
            PhaseRunStatus(phase="ingesta", ok=True, elapsed_seconds=12.5),
            PhaseRunStatus(phase="diarizacion", ok=True, elapsed_seconds=45.3),
        ],
    )


# ──────────────────────────────────────────────────────────
# manifest_to_dict
# ──────────────────────────────────────────────────────────


class TestManifestToDict:
    def test_serializa_campos_basicos(self):
        m = _manifest()
        d = manifest_to_dict(m)
        assert d["run_id"] == "20260706-153012"
        assert d["playlist_name"] == "Play-PoliTest"
        assert d["status"] == "ok"

    def test_serializa_path_como_string(self):
        m = _manifest()
        m.cache_dir = Path("/data/cache")
        d = manifest_to_dict(m)
        assert d["cache_dir"] == "/data/cache"

    def test_path_none_se_omite(self):
        m = _manifest()
        d = manifest_to_dict(m)
        assert "cache_dir" not in d or d["cache_dir"] is None

    def test_serializa_videos(self):
        d = manifest_to_dict(_manifest())
        assert len(d["videos"]) == 2
        assert d["videos"][0]["video_id"] == "v1"
        assert d["videos"][1]["error"] == "HTTP 403"

    def test_serializa_phases(self):
        d = manifest_to_dict(_manifest())
        assert len(d["phases"]) == 2
        assert d["phases"][0]["phase"] == "ingesta"
        assert d["phases"][0]["ok"] is True


# ──────────────────────────────────────────────────────────
# manifest_to_json
# ──────────────────────────────────────────────────────────


class TestManifestToJson:
    def test_produce_json_valido(self):
        j = manifest_to_json(_manifest())
        data = json.loads(j)
        assert data["run_id"] == "20260706-153012"

    def test_es_pretty_printed(self):
        j = manifest_to_json(_manifest())
        assert "\n" in j  # indent != None


# ──────────────────────────────────────────────────────────
# guardar_manifest
# ──────────────────────────────────────────────────────────


class TestGuardarManifest:
    def test_crea_directorio_y_archivo(self, tmp_path: Path):
        m = _manifest()
        output_dir = tmp_path / "output" / "runs"

        path = guardar_manifest(m, output_dir)

        assert path.exists()
        assert path.name == "manifest.json"
        assert path.parent == output_dir / "20260706-153012"

    def test_contenido_es_json_valido(self, tmp_path: Path):
        path = guardar_manifest(_manifest(), tmp_path / "runs")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["playlist_name"] == "Play-PoliTest"

    def test_devuelve_path_absoluto(self, tmp_path: Path):
        path = guardar_manifest(_manifest(), tmp_path / "runs")
        assert path.is_absolute()


# ──────────────────────────────────────────────────────────
# resumen_final
# ──────────────────────────────────────────────────────────


class TestResumenFinal:
    def test_resumen_contiene_run_id(self):
        result = RunResult(manifest=_manifest(), exit_code=0)
        s = resumen_final(result)
        assert "20260706-153012" in s

    def test_resumen_contiene_status(self):
        result = RunResult(manifest=_manifest(status="partial"), exit_code=0)
        s = resumen_final(result)
        assert "partial" in s

    def test_resumen_contiene_contador_videos(self):
        result = RunResult(manifest=_manifest(), exit_code=0)
        s = resumen_final(result)
        assert "1 procesado" in s
        assert "1 omitido" in s

    def test_resumen_contiene_cache_status(self):
        m = _manifest()
        result = RunResult(manifest=m, exit_code=0)
        s = resumen_final(result)
        assert "cache" in s.lower()
