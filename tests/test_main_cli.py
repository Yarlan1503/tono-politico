"""Tests del CLI ligero en main.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from tono_politico.config import Config
from tono_politico.pipeline.models import PhaseRunStatus, RunManifest, RunResult


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
        self.analyze_resume_calls: list[tuple[str, int, str, str | None]] = []
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

    def analyze_resume(
        self,
        run_dir: str,
        topico_id: int,
        tema: str,
        output_path: str | None,
    ) -> RunResult:
        self.analyze_resume_calls.append((run_dir, topico_id, tema, output_path))
        return RunResult(
            manifest=RunManifest(
                run_id="run-001",
                playlist_url="(resume)",
                playlist_name="Play-PoliTest",
                status="ok",
            ),
            exit_code=0,
            informe_path=Path(output_path) if output_path else None,
        )


class FakeFailingRunner(FakeRunner):
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
                status="failed",
                phases=[
                    PhaseRunStatus(
                        phase="filtrado",
                        ok=False,
                        message="No hay segmentos para el tópico 0",
                    )
                ],
            ),
            exit_code=1,
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


def test_main_analyze_imprime_mensaje_de_fallo(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeFailingRunner, raising=False)

    exit_code = cli.main(
        [
            "--playlist",
            "playlist-url",
            "--topico",
            "0",
            "--tema",
            "seguridad",
            "--config",
            str(_config(tmp_path)),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Pipeline falló" in captured.out
    assert "filtrado" in captured.out
    assert "No hay segmentos para el tópico 0" in captured.out


# ============================================================
# Tests CLI extendidos (Task 16)
# ============================================================


def test_topico_sin_tema_produce_error_argparse(monkeypatch, tmp_path: Path):
    """--topico sin --tema debe producir error de argparse."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--playlist",
                "url",
                "--topico",
                "0",
                "--config",
                str(_config(tmp_path)),
            ]
        )
    # argparse.error() llama sys.exit(2)
    assert exc_info.value.code == 2
    # No se debe instanciar el runner
    assert len(FakeRunner.instances) == 0


def test_resume_llama_analyze_resume_no_discover(monkeypatch, tmp_path: Path):
    """--resume debe llamar analyze_resume y no discover."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    exit_code = cli.main(
        [
            "--resume",
            "output/runs/politest-smoke",
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
    assert runner.discover_calls == []
    assert runner.analyze_calls == []
    assert runner.analyze_resume_calls == [
        ("output/runs/politest-smoke", 0, "seguridad", "output/test")
    ]


def test_resume_sin_topico_produce_error(monkeypatch, tmp_path: Path):
    """--resume sin --topico debe producir error de argparse."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--resume",
                "output/runs/run-001",
                "--config",
                str(_config(tmp_path)),
            ]
        )
    assert exc_info.value.code == 2
    assert len(FakeRunner.instances) == 0


def test_playlist_es_opcional_con_resume(monkeypatch, tmp_path: Path):
    """--playlist no es obligatorio cuando se usa --resume."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    exit_code = cli.main(
        [
            "--resume",
            "output/runs/run-001",
            "--topico",
            "1",
            "--tema",
            "economía",
            "--config",
            str(_config(tmp_path)),
        ]
    )

    assert exit_code == 0
    assert len(FakeRunner.instances) == 1
    runner = FakeRunner.instances[0]
    assert runner.analyze_resume_calls == [("output/runs/run-001", 1, "economía", None)]


def test_sin_playlist_y_sin_resume_produce_error(monkeypatch, tmp_path: Path):
    """Sin --playlist y sin --resume debe producir error."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--config", str(_config(tmp_path))])
    assert exc_info.value.code == 2


def test_config_custom_se_carga(monkeypatch, tmp_path: Path):
    """--config debe cargar el YAML desde la ruta especificada."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    config_path = _config(tmp_path)

    cli.main(["--playlist", "url", "--config", str(config_path)])

    assert len(FakeRunner.instances) == 1
    runner = FakeRunner.instances[0]
    assert isinstance(runner.cfg, Config)


def test_keep_cache_llega_al_runner(monkeypatch, tmp_path: Path):
    """--keep-cache debe llegar al PipelineRunner como keep_cache=True."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    cli.main(["--playlist", "url", "--config", str(_config(tmp_path)), "--keep-cache"])

    assert len(FakeRunner.instances) == 1
    runner = FakeRunner.instances[0]
    assert runner.keep_cache is True


def test_sin_keep_cache_llega_false(monkeypatch, tmp_path: Path):
    """Sin --keep-cache el runner recibe keep_cache=False."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    cli.main(["--playlist", "url", "--config", str(_config(tmp_path))])

    assert len(FakeRunner.instances) == 1
    runner = FakeRunner.instances[0]
    assert runner.keep_cache is False


def test_main_retorna_int_no_sys_exit(monkeypatch, tmp_path: Path):
    """main() debe retornar int, no llamar sys.exit directamente."""
    cli = _load_cli_module()
    FakeRunner.instances = []
    monkeypatch.setattr(cli, "PipelineRunner", FakeRunner, raising=False)

    result = cli.main(["--playlist", "url", "--config", str(_config(tmp_path))])

    assert isinstance(result, int)
    assert result == 0
