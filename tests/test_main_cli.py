"""Tests del CLI ligero en main.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from tono_politico.config import Config
from tono_politico.pipeline.models import RunManifest, RunResult


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("tono_politico_main", Path("main.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRunner:
    instances: list[FakeRunner] = []

    def __init__(self, cfg: dict[str, Any], factories: Any, keep_cache: bool = False):
        self.cfg = cfg
        self.factories = factories
        self.keep_cache = keep_cache
        self.discover_calls: list[str] = []
        self.analyze_calls: list[tuple[str, int, str, str | None]] = []
        FakeRunner.instances.append(self)

    def discover(self, playlist_url: str) -> RunResult:
        self.discover_calls.append(playlist_url)
        return RunResult(
            manifest=RunManifest(
                run_id="run-001",
                playlist_url=playlist_url,
                playlist_name="Play-PoliTest",
                status="ok",
            ),
            exit_code=0,
        )

    def analyze(
        self,
        playlist_url: str,
        topico_id: int,
        tema: str,
        output_path: str | None,
    ) -> RunResult:
        self.analyze_calls.append((playlist_url, topico_id, tema, output_path))
        return RunResult(
            manifest=RunManifest(
                run_id="run-001",
                playlist_url=playlist_url,
                playlist_name="Play-PoliTest",
                status="ok",
            ),
            exit_code=0,
            informe_path=Path(output_path) if output_path else None,
        )


def _config(path: Path) -> Path:
    config_path = path / "config.yaml"
    config_path.write_text(
        "project:\n  data_dir: data\ndiarizacion:\n  actor_objetivo: Lilly Téllez\n",
        encoding="utf-8",
    )
    return config_path


def test_main_discover_retorna_exit_code_del_runner(monkeypatch, tmp_path: Path):
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    exit_code = cli.main(
        ["--playlist", "playlist-url", "--config", str(_config(tmp_path)), "--keep-cache"]
    )

    assert exit_code == 0
    assert len(FakeRunner.instances) == 1
    runner = FakeRunner.instances[0]
    assert isinstance(runner.cfg, Config)
    assert runner.keep_cache is True
    assert runner.discover_calls == ["playlist-url"]
    assert runner.analyze_calls == []


def test_main_analyze_retorna_exit_code_del_runner(monkeypatch, tmp_path: Path):
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    exit_code = cli.main(
        [
            "--playlist",
            "playlist-url",
            "--topico",
            "0",
            "--tema",
            "seguridad",
            "--output",
            "output/test",
            "--config",
            str(_config(tmp_path)),
        ]
    )

    assert exit_code == 0
    assert len(FakeRunner.instances) == 1
    runner = FakeRunner.instances[0]
    assert isinstance(runner.cfg, Config)
    assert runner.discover_calls == []
    assert runner.analyze_calls == [("playlist-url", 0, "seguridad", "output/test")]
